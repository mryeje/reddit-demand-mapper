import sqlite3
import json
import re
from sentence_transformers import SentenceTransformer, util
import torch
from collections import defaultdict, Counter
import os
from datetime import datetime
from rapidfuzz import fuzz

class ImprovedVideoMatcher:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Enhanced keyword mapping for better matching
        self.theme_keywords = {
            'cutting_tools': [
                'chainsaw', 'chain saw', 'saw', 'cutting', 'blade', 'cut', 'miter saw', 
                'circular saw', 'reciprocating saw', 'jigsaw', 'table saw', 'band saw'
            ],
            'drilling_tools': [
                'drill', 'drilling', 'impact driver', 'hammer drill', 'driver', 
                'drill bit', 'boring', 'hole', 'impact', 'torque'
            ],
            'cleaning_equipment': [
                'pressure washer', 'power washer', 'washer', 'cleaning', 'wash', 
                'clean', 'pressure wash', 'power wash', 'hose', 'nozzle'
            ],
            'lawn_care': [
                'lawn mower', 'mower', 'grass', 'lawn', 'yard', 'trimmer', 
                'string trimmer', 'weed eater', 'edger', 'leaf blower', 'blower'
            ],
            'kitchen_appliances': [
                'microwave', 'oven', 'refrigerator', 'dishwasher', 'air fryer', 
                'blender', 'mixer', 'toaster', 'coffee maker', 'food processor'
            ],
            'repair_help': [
                'repair', 'fix', 'broken', 'maintenance', 'troubleshoot', 
                'replace', 'replacement', 'part', 'service', 'diagnose'
            ],
            'beginner_guidance': [
                'beginner', 'guide', 'tutorial', 'how to', 'basic', 'introduction', 
                'getting started', 'first time', 'learn', 'education'
            ],
            'budget_options': [
                'budget', 'cheap', 'affordable', 'value', 'deal', 'sale', 
                'discount', 'price', 'cost', 'inexpensive'
            ]
        }
        
        # Content type mapping for better video selection
        self.content_preferences = {
            'cutting_tools': ['review', 'comparison', 'guide', 'tips', 'maintenance'],
            'drilling_tools': ['comparison', 'review', 'tutorial', 'guide'],
            'cleaning_equipment': ['tutorial', 'tips', 'review', 'demonstration'],
            'lawn_care': ['review', 'guide', 'tips', 'comparison'],
            'kitchen_appliances': ['review', 'comparison', 'tips', 'guide'],
            'repair_help': ['repair', 'fix', 'tutorial', 'maintenance'],
            'beginner_guidance': ['guide', 'tutorial', 'basics', 'introduction'],
            'budget_options': ['review', 'comparison', 'budget', 'value']
        }

        # New: synonym mapping for fuzzy matching
        self.synonyms = {
            "oven": ["range", "stove"],
            "leveling foot": ["levelling leg", "front foot", "rear leveling leg", "adjustable leg", "foot leveller"],
            "washer": ["washing machine", "laundry machine"],
            "dryer": ["clothes dryer"],
            "refrigerator": ["fridge", "freezer"],
            "pressure washer": ["power washer"],
        }

    def load_data(self, reddit_json_path, youtube_db_paths):
        """Load Reddit themes and YouTube videos from multiple databases"""
        # Load Reddit data
        with open(reddit_json_path, "r", encoding="utf-8") as f:
            self.themes = json.load(f)

        self.yt_videos = []
        
        # Loop over all DBs provided
        for db_path in youtube_db_paths:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT video_id, title, description FROM videos")
            videos = [
                {
                    "video_id": row[0],
                    "title": row[1],
                    "description": row[2] or "",
                    "channel": os.path.splitext(os.path.basename(db_path))[0]
                }
                for row in cursor.fetchall()
            ]
            conn.close()
            
            print(f"Loaded {len(videos)} videos from {db_path}")
            self.yt_videos.extend(videos)

        print(f"Total videos loaded: {len(self.yt_videos)}")
        print(f"Loaded {len(self.themes)} Reddit themes")

    def expand_keywords(self, text):
        text_lower = text.lower()
        expanded = [text_lower]
        for key, syns in self.synonyms.items():
            if key in text_lower:
                expanded.extend(syns)
            else:
                for s in syns:
                    if s in text_lower:
                        expanded.append(key)
                        expanded.extend([syn for syn in syns if syn != s])
        return expanded

    def extract_reddit_keywords(self, theme_data):
        """Extract meaningful keywords from Reddit posts in a theme"""
        all_text = ""
        
        # Combine title and text from all sample posts
        for post in theme_data.get('sample_posts', []):
            all_text += post.get('title', '') + " " + post.get('selftext', '') + " "
        
        # Add matched keywords from the original analysis
        matched_keywords = []
        for post in theme_data.get('sample_posts', []):
            matched_keywords.extend(post.get('matched_keywords', []))
        
        all_text += " " + " ".join(matched_keywords)
        
        # Extract tool/brand names and important terms
        tool_pattern = r'\b(?:dewalt|makita|milwaukee|ryobi|bosch|craftsman|black decker|porter cable|metabo|festool|ridgid)\b'
        product_pattern = r'\b(?:drill|saw|grinder|sander|router|planer|impact|driver|battery|charger|blade|bit)\b'
        
        brands = re.findall(tool_pattern, all_text.lower())
        products = re.findall(product_pattern, all_text.lower())
        
        return {
            'full_text': all_text.lower(),
            'brands': list(set(brands)),
            'products': list(set(products)),
            'matched_keywords': list(set(matched_keywords))
        }

    def score_video_relevance(self, video, theme_name, reddit_keywords):
        """Score how relevant a video is to a Reddit theme"""
        video_text = (video['title'] + " " + video['description']).lower()
        score = 0
        reasons = []
        
        # 1. Exact keyword matches (highest priority)
        theme_kw = self.theme_keywords.get(theme_name, [])
        for keyword in theme_kw:
            if keyword in video_text:
                score += 10
                reasons.append(f"Theme keyword: {keyword}")
            else:
                # Fuzzy match
                fuzzy_score = fuzz.partial_ratio(keyword, video_text)
                if fuzzy_score >= 70:
                    score += 8
                    reasons.append(f"Fuzzy theme keyword match: {keyword} ({fuzzy_score})")
        
        # 2. Brand matches
        for brand in reddit_keywords['brands']:
            if brand in video_text:
                score += 8
                reasons.append(f"Brand match: {brand}")
        
        # 3. Product matches
        for product in reddit_keywords['products']:
            if product in video_text:
                score += 6
                reasons.append(f"Product match: {product}")
        
        # 4. Reddit-specific keywords (with fuzzy)
        for keyword in reddit_keywords['matched_keywords']:
            if keyword.lower() in video_text:
                score += 5
                reasons.append(f"Reddit keyword: {keyword}")
            else:
                fuzzy_score = fuzz.partial_ratio(keyword.lower(), video_text)
                if fuzzy_score >= 75:
                    score += 4
                    reasons.append(f"Fuzzy Reddit keyword: {keyword} ({fuzzy_score})")
        
        # 5. Content type preference
        preferred_content = self.content_preferences.get(theme_name, [])
        for content_type in preferred_content:
            if content_type in video_text:
                score += 3
                reasons.append(f"Content type: {content_type}")
        
        # 6. Avoid pure repair videos for non-repair themes
        if theme_name != 'repair_help':
            repair_indicators = ['replacing', 'replacement part', 'repair -', 'part #', 'part number']
            repair_count = sum(1 for indicator in repair_indicators if indicator in video_text)
            if repair_count >= 2:
                score -= 5
                reasons.append("Heavy repair focus (penalty)")
        
        return score, reasons

    def find_best_videos(self, theme_name, theme_data, top_k=3):
        """Find the most relevant videos for a theme"""
        reddit_keywords = self.extract_reddit_keywords(theme_data)
        
        # Score all videos
        video_scores = []
        for video in self.yt_videos:
            score, reasons = self.score_video_relevance(video, theme_name, reddit_keywords)
            if score > 0:  # Only include videos with positive relevance
                video_scores.append({
                    'video': video,
                    'score': score,
                    'reasons': reasons
                })
        
        # Sort by score and return top videos
        video_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # If we have few high-scoring videos, use semantic similarity as backup
        if len(video_scores) < top_k:
            semantic_videos = self.semantic_fallback(theme_data, reddit_keywords, top_k)
            video_scores.extend(semantic_videos)
        
        return video_scores[:top_k]

    def semantic_fallback(self, theme_data, reddit_keywords, top_k):
        """Use semantic similarity when keyword matching fails"""
        theme_texts = []
        for post in theme_data.get('sample_posts', []):
            theme_texts.append(post.get('title', '') + " " + post.get('selftext', ''))
        
        if not theme_texts:
            return []
        
        theme_text = " ".join(theme_texts)
        theme_embedding = self.model.encode([theme_text], convert_to_tensor=True)
        
        yt_texts = [v["title"] + " " + v["description"] for v in self.yt_videos]
        yt_embeddings = self.model.encode(yt_texts, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(theme_embedding, yt_embeddings)[0]
        top_idx = torch.topk(cosine_scores, k=min(top_k, len(cosine_scores))).indices.tolist()
        
        semantic_matches = []
        for idx in top_idx:
            if cosine_scores[idx] > 0.3:  # Minimum similarity threshold
                semantic_matches.append({
                    'video': self.yt_videos[idx],
                    'score': float(cosine_scores[idx]) * 5,
                    'reasons': [f'Semantic similarity: {cosine_scores[idx]:.3f}']
                })
        
        return semantic_matches

    def process_all_themes(self, output_path=None, top_k=3):
        """Process all themes and add matched videos"""
        print("Processing themes...")

        for i, theme in enumerate(self.themes):
            theme_name = theme.get('theme', 'unknown')
            print(f"Processing theme {i+1}/{len(self.themes)}: {theme_name}")

            best_videos = self.find_best_videos(theme_name, theme, top_k)

            matched_videos = []
            for video_match in best_videos:
                video = video_match['video']
                matched_videos.append({
                    "video_id": video["video_id"],
                    "title": video["title"],
                    "channel": video["channel"],
                    "description_snippet": (
                        video["description"][:200] + "..."
                        if len(video["description"]) > 200 else video["description"]
                    ),
                    "relevance_score": video_match['score'],
                    "match_reasons": video_match['reasons']
                })

            theme['matched_videos'] = matched_videos

            if matched_videos:
                print(f"  → Found {len(matched_videos)} videos (scores: {[v['relevance_score'] for v in matched_videos]})")
            else:
                print("  → No relevant videos found")

        # Save if requested
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.themes, f, indent=2, ensure_ascii=False)
            print(f"Enhanced data saved to {output_path}")

        self.print_summary()

        # Return enriched themes for external saving
        return self.themes

    def print_summary(self):
        print("\n" + "="*50)
        print("MATCHING SUMMARY")
        print("="*50)
        
        total_themes = len(self.themes)
        themes_with_videos = sum(1 for theme in self.themes if theme.get('matched_videos'))
        
        print(f"Total themes processed: {total_themes}")
        print(f"Themes with matched videos: {themes_with_videos}")
        print(f"Success rate: {themes_with_videos/total_themes*100:.1f}%")
        
        print("\nPer-theme results:")
        for theme in self.themes:
            theme_name = theme.get('theme', 'unknown')
            video_count = len(theme.get('matched_videos', []))
            if video_count > 0:
                best_score = max(v.get('relevance_score', 0) for v in theme['matched_videos'])
                print(f"  {theme_name:20} → {video_count} videos (best score: {best_score:.1f})")
            else:
                print(f"  {theme_name:20} → No matches")

def main():
    REDDIT_JSON_PATH = "reports/latest.json"
    YOUTUBE_DB_PATHS = [
        "fixtechguides.db",
        "partselect.db"
    ]
    TOP_K = 3

    matcher = ImprovedVideoMatcher()
    matcher.load_data(REDDIT_JSON_PATH, YOUTUBE_DB_PATHS)

    os.makedirs("reports", exist_ok=True)

    # File paths
    latest_path = os.path.join("reports", "latest.json")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = os.path.join("reports", f"reddit_with_videos_{timestamp}.json")

    # Get enriched data
    data = matcher.process_all_themes(top_k=TOP_K)

    # Save latest.json
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Save timestamped archive
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Enriched results saved to:\n- {latest_path}\n- {archive_path}")

if __name__ == "__main__":
    main()

