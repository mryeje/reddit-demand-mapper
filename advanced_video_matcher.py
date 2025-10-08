import sqlite3
import json
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from collections import defaultdict, Counter
import spacy
from datetime import datetime
import os
import time
import gc
import psutil
import warnings

class AdvancedVideoMatcher:
    def __init__(self, force_cpu=False, verbose=True):
        """
        Initialize the Advanced Video Matcher with automatic device detection
        
        Args:
            force_cpu (bool): Force CPU usage even if GPU is available
            verbose (bool): Print device and performance information
        """
        self.verbose = verbose
        self.device = self._setup_device(force_cpu)
        self.performance_stats = {
            'initialization_time': time.time(),
            'embedding_times': [],
            'total_videos_processed': 0,
            'memory_peak': 0
        }
        
        if self.verbose:
            self._print_system_info()
        
        # Initialize models with automatic device selection
        self._initialize_models()
        
        # Initialize spaCy for NER
        self._initialize_spacy()
        
        # Intent patterns and video content types (same as before)
        self._setup_patterns()
        
        if self.verbose:
            init_time = time.time() - self.performance_stats['initialization_time']
            print(f"🚀 Initialization complete in {init_time:.2f}s")

    def _setup_device(self, force_cpu):
        """Setup computing device with automatic GPU detection"""
        if force_cpu:
            device = 'cpu'
            reason = "Forced CPU mode"
        elif not torch.cuda.is_available():
            device = 'cpu' 
            reason = "CUDA not available"
        elif os.getenv('GITHUB_ACTIONS'):
            device = 'cpu'
            reason = "GitHub Actions environment detected"
        else:
            device = 'cuda'
            reason = f"GPU detected: {torch.cuda.get_device_name(0)}"
        
        if self.verbose:
            print(f"🔧 Device selection: {device.upper()} ({reason})")
            
        return device

    def _print_system_info(self):
        """Print system information for debugging"""
        print("="*60)
        print("ADVANCED VIDEO MATCHER - SYSTEM INFO")
        print("="*60)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # System memory info
        memory = psutil.virtual_memory()
        print(f"System RAM: {memory.total / 1e9:.1f}GB (Available: {memory.available / 1e9:.1f}GB)")
        print(f"CPU cores: {psutil.cpu_count()}")
        print("="*60)

    def _initialize_models(self):
        """Initialize sentence transformer models with device selection"""
        try:
            if self.verbose:
                print("📥 Loading semantic models...")
            
            # Primary semantic model
            model_start = time.time()
            self.semantic_model = SentenceTransformer(
                'all-MiniLM-L6-v2', 
                device=self.device
            )
            
            # Secondary intent model (using same model for efficiency)
            self.intent_model = self.semantic_model
            
            model_time = time.time() - model_start
            if self.verbose:
                print(f"✅ Models loaded in {model_time:.2f}s on {self.device.upper()}")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("🔄 Falling back to CPU...")
            self.device = 'cpu'
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            self.intent_model = self.semantic_model

    def _initialize_spacy(self):
        """Initialize spaCy with error handling"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            if self.verbose:
                print("✅ spaCy English model loaded")
        except IOError:
            warnings.warn(
                "spaCy English model not found. Install with: python -m spacy download en_core_web_sm\n"
                "Named Entity Recognition will be disabled."
            )
            self.nlp = None

    def _setup_patterns(self):
        """Setup intent patterns and video content types"""
        self.intent_patterns = {
            'buying_advice': {
                'patterns': [
                    r'\b(which|what|best|recommend|should i buy|worth it|opinions on)\b',
                    r'\b(buying guide|purchase|shopping for|looking for)\b',
                    r'\b(budget|price|cost|cheap|expensive|value)\b'
                ],
                'weight': 1.0
            },
            'troubleshooting': {
                'patterns': [
                    r'\b(broken|not working|problem|issue|help|trouble|fix)\b',
                    r'\b(error|fault|malfunction|stopped|died|failed)\b',
                    r'\b(why won\'t|can\'t get|won\'t start|won\'t turn)\b'
                ],
                'weight': 1.2
            },
            'how_to_guide': {
                'patterns': [
                    r'\b(how to|how do i|tutorial|guide|instructions|steps)\b',
                    r'\b(learn|teach|show me|explain|walkthrough)\b',
                    r'\b(beginner|new to|first time|getting started)\b'
                ],
                'weight': 1.1
            },
            'comparison': {
                'patterns': [
                    r'\b(vs|versus|compare|comparison|difference between)\b',
                    r'\b(better|superior|which is|pros and cons)\b',
                    r'\b(alternative|substitute|replacement)\b'
                ],
                'weight': 0.9
            },
            'maintenance': {
                'patterns': [
                    r'\b(maintain|maintenance|care|service|clean|oil)\b',
                    r'\b(upkeep|preserve|extend life|last longer)\b',
                    r'\b(schedule|routine|regular|periodic)\b'
                ],
                'weight': 0.8
            }
        }
        
        self.video_content_types = {
            'review': {
                'indicators': ['review', 'honest', 'opinion', 'pros', 'cons', 'worth it', 'after', 'months', 'years'],
                'boost': 1.2
            },
            'tutorial': {
                'indicators': ['how to', 'step by step', 'guide', 'tutorial', 'diy', 'instructions', 'learn'],
                'boost': 1.3
            },
            'comparison': {
                'indicators': ['vs', 'versus', 'compare', 'comparison', 'best', 'top', 'battle'],
                'boost': 1.1
            },
            'unboxing': {
                'indicators': ['unbox', 'unboxing', 'first look', 'initial', 'new', 'out of box'],
                'boost': 0.7
            },
            'repair': {
                'indicators': ['repair', 'fix', 'broken', 'replace', 'troubleshoot', 'diagnose'],
                'boost': 1.0
            }
        }

    def _monitor_memory(self):
        """Monitor memory usage for performance tracking"""
        if self.device == 'cuda':
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            self.performance_stats['memory_peak'] = max(
                self.performance_stats['memory_peak'], 
                gpu_memory
            )
        
        system_memory = psutil.virtual_memory().used / 1e9
        self.performance_stats['memory_peak'] = max(
            self.performance_stats['memory_peak'],
            system_memory
        )

    def _cleanup_memory(self):
        """Clean up memory to prevent OOM issues"""
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def load_data(self, reddit_json_path, youtube_db_paths):
        """Load Reddit themes and YouTube videos with performance monitoring"""
        load_start = time.time()
        
        # Load Reddit data
        with open(reddit_json_path, "r", encoding="utf-8") as f:
            self.themes = json.load(f)

        self.yt_videos = []
        
        # Load YouTube videos from multiple databases
        for db_path in youtube_db_paths:
            if not os.path.exists(db_path):
                print(f"⚠️  Warning: Database not found: {db_path}")
                continue
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT video_id, title, description FROM videos")
            videos = [
                {
                    "video_id": row[0],
                    "title": row[1],
                    "description": row[2] or "",
                    "channel": os.path.splitext(os.path.basename(db_path))[0],
                    "full_text": self.preprocess_text(row[1] + " " + (row[2] or ""))
                }
                for row in cursor.fetchall()
            ]
            conn.close()
            
            if self.verbose:
                print(f"📦 Loaded {len(videos)} videos from {db_path}")
            self.yt_videos.extend(videos)

        load_time = time.time() - load_start
        
        if self.verbose:
            print(f"📊 Data loading complete:")
            print(f"   Videos: {len(self.yt_videos):,}")
            print(f"   Themes: {len(self.themes):,}")
            print(f"   Load time: {load_time:.2f}s")

    def preprocess_text(self, text):
        """Advanced text preprocessing with performance optimization"""
        if not text:
            return ""
        
        text = text.lower()
        
        # Batch normalize common variations
        normalizations = {
            r'\b(drill|drills|drilling)\b': 'drill',
            r'\b(saw|saws|sawing)\b': 'saw',
            r'\b(pressure\s*wash|power\s*wash)\b': 'pressure wash',
            r'\b(weed\s*eater|string\s*trimmer)\b': 'string trimmer',
            r'\b(lawn\s*mower|grass\s*cutter)\b': 'lawn mower',
            r'\b(air\s*fryer|airfryer)\b': 'air fryer',
            r'\b(impact\s*driver|impact\s*drill)\b': 'impact driver'
        }
        
        for pattern, replacement in normalizations.items():
            text = re.sub(pattern, replacement, text)
        
        # Handle model numbers and part numbers
        text = re.sub(r'\b[a-z]*\d+[a-z]*\b', '[MODEL]', text)
        
        # Clean but preserve important punctuation
        text = re.sub(r'[^\w\s\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def extract_dynamic_features(self, theme_data):
        """Extract features dynamically from Reddit posts"""
        # Combine all text from posts
        all_texts = []
        for post in theme_data.get('sample_posts', []):
            title = post.get('title', '')
            selftext = post.get('selftext', '')
            combined = self.preprocess_text(title + " " + selftext)
            if combined:
                all_texts.append(combined)
        
        if not all_texts:
            return {}
        
        # TF-IDF analysis for important terms
        try:
            vectorizer = TfidfVectorizer(
                max_features=30,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_terms = [(feature_names[i], mean_scores[i]) for i in np.argsort(mean_scores)[::-1]]
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  TF-IDF extraction failed: {e}")
            top_terms = []
        
        # Named Entity Recognition
        entities = []
        brands = []
        
        if self.nlp:
            for text in all_texts:
                try:
                    doc = self.nlp(text)
                    for ent in doc.ents:
                        if ent.label_ in ['ORG', 'PRODUCT']:
                            entities.append(ent.text.lower())
                        if ent.label_ == 'ORG':
                            brands.append(ent.text.lower())
                except Exception as e:
                    continue  # Skip problematic texts
        
        # Extract specific product mentions
        products = []
        product_patterns = [
            r'\b(dewalt|makita|milwaukee|ryobi|bosch|craftsman|black\s*decker)\b',
            r'\b(chainsaw|drill|saw|grinder|mower|washer|fryer|blender)\b'
        ]
        
        for text in all_texts:
            for pattern in product_patterns:
                matches = re.findall(pattern, text)
                products.extend(matches)
        
        return {
            'tfidf_terms': [term for term, score in top_terms[:15]],
            'entities': list(set(entities)),
            'brands': list(set(brands)),
            'products': list(set(products)),
            'combined_text': " ".join(all_texts)
        }

    def analyze_intent(self, text):
        """Analyze user intent from text"""
        intent_scores = {}
        
        for intent, config in self.intent_patterns.items():
            score = 0
            for pattern in config['patterns']:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            
            intent_scores[intent] = score * config['weight']
        
        return intent_scores

    def classify_video_content(self, video):
        """Classify video content type"""
        text = video['title'].lower() + " " + video['description'].lower()
        
        type_scores = {}
        for content_type, config in self.video_content_types.items():
            score = 0
            for indicator in config['indicators']:
                if indicator in text:
                    score += 1
            type_scores[content_type] = score
        
        primary_type = max(type_scores, key=type_scores.get) if any(type_scores.values()) else 'general'
        return primary_type, type_scores

    def calculate_semantic_similarity(self, reddit_features, video):
        """Calculate semantic similarity with performance optimization"""
        reddit_text = reddit_features['combined_text']
        video_text = video['full_text']
        
        if not reddit_text or not video_text:
            return 0.0
        
        try:
            embedding_start = time.time()
            
            # Encode texts with batch processing for efficiency
            texts = [reddit_text, video_text]
            embeddings = self.semantic_model.encode(
                texts, 
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=32  # Optimize batch size for your hardware
            )
            
            # Calculate cosine similarity
            similarity = util.cos_sim(embeddings[0:1], embeddings[1:2])[0][0].item()
            
            embedding_time = time.time() - embedding_start
            self.performance_stats['embedding_times'].append(embedding_time)
            
            return max(0, similarity)  # Ensure non-negative
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Semantic similarity calculation failed: {e}")
            return 0.0

    def calculate_keyword_overlap(self, reddit_features, video):
        """Calculate enhanced keyword overlap score"""
        video_text = video['full_text']
        score = 0
        matches = []
        
        # TF-IDF terms (high weight)
        for term in reddit_features['tfidf_terms']:
            if term in video_text:
                score += 5
                matches.append(f"tfidf:{term}")
            else:
                # Fuzzy matching for TF-IDF terms
                fuzzy_score = max(fuzz.partial_ratio(term, video_text), 
                                fuzz.token_sort_ratio(term, video_text))
                if fuzzy_score >= 75:
                    score += 3
                    matches.append(f"fuzzy_tfidf:{term}({fuzzy_score})")
        
        # Brands (medium-high weight)
        for brand in reddit_features['brands']:
            if brand in video_text:
                score += 4
                matches.append(f"brand:{brand}")
        
        # Products (medium weight)
        for product in reddit_features['products']:
            if product in video_text:
                score += 3
                matches.append(f"product:{product}")
        
        # Entities (lower weight)
        for entity in reddit_features['entities']:
            if entity in video_text:
                score += 2
                matches.append(f"entity:{entity}")
        
        return score, matches

    def calculate_intent_alignment(self, reddit_features, video):
        """Calculate how well video content aligns with Reddit user intent"""
        reddit_text = reddit_features['combined_text']
        video_text = video['full_text']
        
        reddit_intents = self.analyze_intent(reddit_text)
        video_intents = self.analyze_intent(video_text)
        
        # Calculate intent overlap
        alignment_score = 0
        intent_matches = []
        
        for intent, reddit_score in reddit_intents.items():
            if reddit_score > 0:
                video_score = video_intents.get(intent, 0)
                if video_score > 0:
                    alignment_score += min(reddit_score, video_score) * 2
                    intent_matches.append(f"{intent}:{reddit_score}↔{video_score}")
        
        return alignment_score, intent_matches

    def calculate_content_type_boost(self, video, reddit_features):
        """Apply content type boost based on what users are looking for"""
        video_type, type_scores = self.classify_video_content(video)
        
        # Determine what content type would be most helpful for Reddit users
        reddit_text = reddit_features['combined_text']
        reddit_intents = self.analyze_intent(reddit_text)
        
        # Map intents to preferred content types
        intent_to_content = {
            'buying_advice': ['review', 'comparison'],
            'troubleshooting': ['repair', 'tutorial'],
            'how_to_guide': ['tutorial'],
            'comparison': ['comparison', 'review'],
            'maintenance': ['tutorial', 'repair']
        }
        
        boost = 1.0
        preferred_types = []
        
        for intent, score in reddit_intents.items():
            if score > 0:
                preferred_types.extend(intent_to_content.get(intent, []))
        
        if video_type in preferred_types:
            boost = self.video_content_types[video_type]['boost']
        
        return boost, video_type

    def score_video_relevance(self, video, reddit_features):
        """Comprehensive video relevance scoring with performance monitoring"""
        self._monitor_memory()
        
        # 1. Semantic similarity (40% weight)
        semantic_score = self.calculate_semantic_similarity(reddit_features, video) * 40
        
        # 2. Keyword overlap (30% weight)
        keyword_score, keyword_matches = self.calculate_keyword_overlap(reddit_features, video)
        keyword_score = min(keyword_score, 30)  # Cap at 30 points
        
        # 3. Intent alignment (20% weight)
        intent_score, intent_matches = self.calculate_intent_alignment(reddit_features, video)
        intent_score = min(intent_score, 20)  # Cap at 20 points
        
        # 4. Content type boost (10% weight)
        content_boost, content_type = self.calculate_content_type_boost(video, reddit_features)
        boosted_score = (semantic_score + keyword_score + intent_score) * content_boost
        
        # Quality filters
        quality_penalty = 0
        if len(video['title']) < 10:
            quality_penalty += 5
        if not video['description']:
            quality_penalty += 3
        
        final_score = max(0, boosted_score - quality_penalty)
        
        self.performance_stats['total_videos_processed'] += 1
        
        return {
            'total_score': final_score,
            'semantic_score': semantic_score,
            'keyword_score': keyword_score,
            'intent_score': intent_score,
            'content_boost': content_boost,
            'content_type': content_type,
            'keyword_matches': keyword_matches,
            'intent_matches': intent_matches,
            'quality_penalty': quality_penalty
        }

    def find_best_videos(self, theme_name, theme_data, top_k=5):
        """Find the most relevant videos using the advanced algorithm with batch processing"""
        if self.verbose:
            print(f"🔍 Processing theme: {theme_name}")
        
        # Extract features from Reddit posts
        reddit_features = self.extract_dynamic_features(theme_data)
        if not reddit_features['combined_text']:
            if self.verbose:
                print(f"  ⚠️  No valid text found for theme")
            return []
        
        # Process videos in batches for memory efficiency
        batch_size = 1000 if self.device == 'cuda' else 500
        video_scores = []
        
        for i in range(0, len(self.yt_videos), batch_size):
            batch = self.yt_videos[i:i+batch_size]
            
            for video in batch:
                score_details = self.score_video_relevance(video, reddit_features)
                
                if score_details['total_score'] > 5:  # Minimum threshold
                    video_scores.append({
                        'video': video,
                        'score_details': score_details,
                        'total_score': score_details['total_score']
                    })
            
            # Clean up memory after each batch
            if i % (batch_size * 5) == 0:  # Every 5 batches
                self._cleanup_memory()
        
        # Sort by total score
        video_scores.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Apply diversity filter to avoid too many videos from same channel
        diverse_results = self.apply_diversity_filter(video_scores, top_k)
        
        if self.verbose:
            print(f"  ✅ Found {len(diverse_results)} relevant videos")
            if diverse_results:
                avg_score = sum(v['total_score'] for v in diverse_results) / len(diverse_results)
                print(f"     Average score: {avg_score:.1f}")
        
        return diverse_results

    def apply_diversity_filter(self, scored_videos, top_k):
        """Ensure diversity in results (max 2 videos per channel)"""
        channel_counts = defaultdict(int)
        diverse_results = []
        
        for video_data in scored_videos:
            channel = video_data['video']['channel']
            if channel_counts[channel] < 2 and len(diverse_results) < top_k:
                diverse_results.append(video_data)
                channel_counts[channel] += 1
        
        # If we still need more videos and have room, add remaining highest-scored ones
        if len(diverse_results) < top_k:
            for video_data in scored_videos:
                if video_data not in diverse_results and len(diverse_results) < top_k:
                    diverse_results.append(video_data)
        
        return diverse_results

    def process_all_themes(self, output_path=None, top_k=5):
        """Process all themes with the advanced matcher and performance tracking"""
        process_start = time.time()
        
        if self.verbose:
            print("🚀 Starting advanced video matching...")
            print(f"   Device: {self.device.upper()}")
            print(f"   Themes to process: {len(self.themes)}")
            print(f"   Videos available: {len(self.yt_videos):,}")
        
        for i, theme in enumerate(self.themes):
            theme_name = theme.get('theme', 'unknown')
            
            if self.verbose and i % 5 == 0:  # Progress update every 5 themes
                progress = (i / len(self.themes)) * 100
                print(f"📊 Progress: {progress:.1f}% ({i}/{len(self.themes)} themes)")
            
            best_videos = self.find_best_videos(theme_name, theme, top_k)
            
            # Format results
            matched_videos = []
            for video_match in best_videos:
                video = video_match['video']
                score_details = video_match['score_details']
                
                matched_videos.append({
                    "video_id": video["video_id"],
                    "title": video["title"],
                    "channel": video["channel"],
                    "description_snippet": (
                        video["description"][:200] + "..."
                        if len(video["description"]) > 200 else video["description"]
                    ),
                    "relevance_score": round(score_details['total_score'], 2),
                    "scoring_breakdown": {
                        "semantic_score": round(score_details['semantic_score'], 2),
                        "keyword_score": score_details['keyword_score'],
                        "intent_score": score_details['intent_score'],
                        "content_boost": round(score_details['content_boost'], 2),
                        "content_type": score_details['content_type']
                    },
                    "match_evidence": {
                        "keyword_matches": score_details['keyword_matches'][:5],
                        "intent_matches": score_details['intent_matches']
                    }
                })
            
            theme['matched_videos'] = matched_videos
        
        # Save results
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.themes, f, indent=2, ensure_ascii=False)
            if self.verbose:
                print(f"💾 Results saved to {output_path}")
        
        process_time = time.time() - process_start
        self.performance_stats['total_process_time'] = process_time
        
        self.print_advanced_summary()
        return self.themes

    def print_advanced_summary(self):
        """Print detailed summary with performance metrics"""
        print("\n" + "="*80)
        print("ADVANCED VIDEO MATCHING SUMMARY")
        print("="*80)
        
        total_themes = len(self.themes)
        themes_with_videos = sum(1 for theme in self.themes if theme.get('matched_videos'))
        
        print(f"📊 Processing Results:")
        print(f"   Total themes processed: {total_themes:,}")
        print(f"   Themes with matched videos: {themes_with_videos:,}")
        print(f"   Success rate: {themes_with_videos/total_themes*100:.1f}%")
        print(f"   Total videos processed: {self.performance_stats['total_videos_processed']:,}")
        
        # Performance metrics
        if self.performance_stats.get('total_process_time'):
            print(f"\n⏱️  Performance Metrics:")
            print(f"   Total processing time: {self.performance_stats['total_process_time']:.1f}s")
            print(f"   Average time per theme: {self.performance_stats['total_process_time']/total_themes:.2f}s")
            
            if self.performance_stats['embedding_times']:
                avg_embedding_time = np.mean(self.performance_stats['embedding_times'])
                print(f"   Average embedding time: {avg_embedding_time*1000:.1f}ms")
                
            if self.device == 'cuda':
                print(f"   Peak GPU memory: {self.performance_stats['memory_peak']:.1f}GB")
        
        # Scoring distribution
        all_scores = []
        for theme in self.themes:
            for video in theme.get('matched_videos', []):
                all_scores.append(video['relevance_score'])
        
        if all_scores:
            print(f"\n📈 Score Distribution:")
            print(f"   Total matched videos: {len(all_scores):,}")
            print(f"   Average score: {np.mean(all_scores):.2f}")
            print(f"   Median score: {np.median(all_scores):.2f}")
            print(f"   Score range: {min(all_scores):.1f} - {max(all_scores):.1f}")
        
        print(f"\n🏆 Top Performing Themes:")
        theme_scores = []
        for theme in self.themes:
            videos = theme.get('matched_videos', [])
            if videos:
                avg_score = sum(v['relevance_score'] for v in videos) / len(videos)
                theme_scores.append((theme.get('theme', 'unknown'), avg_score, len(videos)))
        
        theme_scores.sort(key=lambda x: x[1], reverse=True)
        for i, (theme_name, avg_score, count) in enumerate(theme_scores[:10]):
            print(f"   {i+1:2}. {theme_name:25} → {count} videos (avg: {avg_score:.1f})")
        
        print("="*80)

def main():
    """Main execution function with environment detection"""
    # Configuration
    REDDIT_JSON_PATH = "reports/latest.json"
    YOUTUBE_DB_PATHS = [
        "fixtechguides.db",
        "partselect.db"
    ]
    TOP_K = 5
    
    # Detect environment and adjust settings
    is_github_actions = bool(os.getenv('GITHUB_ACTIONS'))
    force_cpu = is_github_actions or ('--cpu' in os.sys.argv)
    verbose = not is_github_actions  # Less verbose in CI
    
    print("🤖 Advanced Video Matcher")
    print(f"Environment: {'GitHub Actions' if is_github_actions else 'Local'}")
    
    # Initialize the matcher
    matcher = AdvancedVideoMatcher(force_cpu=force_cpu, verbose=verbose)
    
    try:
        # Load data
        if not os.path.exists(REDDIT_JSON_PATH):
            print(f"❌ Reddit data not found: {REDDIT_JSON_PATH}")
            return
        
        matcher.load_data(REDDIT_JSON_PATH, YOUTUBE_DB_PATHS)
        
        # Create reports directory
        os.makedirs("reports", exist_ok=True)
        
        # Process all themes
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/advanced_matched_videos_{timestamp}.json"
        
        enriched_data = matcher.process_all_themes(output_path, TOP_K)
        
        # Also update latest.json
        latest_path = "reports/latest.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Advanced matching complete!")
        print(f"📁 Results saved to: {output_path}")
        print(f"📁 Latest results: {latest_path}")
        
        # Final cleanup
        matcher._cleanup_memory()
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()