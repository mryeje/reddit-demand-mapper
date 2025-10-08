import praw
import csv
import json
import os
import re
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class ImprovedRedditDemandMapper:
    def __init__(self, persist_subreddits=False):
        # Load Reddit credentials
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = 'DemandMapper/2.0'

        if not client_id or not client_secret:
            client_id = 'sDahoUx4K4Bmx9ebIGVtHQ'
            client_secret = 'jUn8xcFBOeh7jBX5eyW4s7JDpSNalQ'
            print("WARNING: Using hard-coded Reddit credentials for local testing.")

        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Focused subreddits for your niches
        self.target_subreddits = [
            # Power Tools
            'Tools', 'Dewalt', 'Makita', 'Milwaukee', 'Ryobi', 'Bosch',
            'woodworking', 'Carpentry', 'HomeImprovement', 'DIY',
            'electricians', 'Plumbing', 'Construction',
            
            # Outdoor Power Equipment
            'landscaping', 'lawncare', 'Chainsaw', 'powerwashing',
            'gardening', 'TreeClimbing', 'arborists',
            
            # Appliances (non-cooking focus)
            'appliances', 'BuyItForLife', 'fixit', 'HomeAppliances',
            'appliancerepair', 'HVAC', 'Plumbing'
        ]
        
        # Your specific niche definitions
        self.niche_categories = {
            'outdoor_power_equipment': {
                'brands': [
                    'stihl', 'husqvarna', 'echo', 'craftsman', 'poulan', 'worx',
                    'ego', 'greenworks', 'ryobi', 'dewalt', 'milwaukee', 'makita'
                ],
                'products': [
                    'chainsaw', 'leaf blower', 'hedge trimmer', 'string trimmer', 
                    'weed eater', 'edger', 'pressure washer', 'power washer',
                    'lawn mower', 'zero turn', 'riding mower', 'push mower',
                    'walk behind', 'self propelled', 'reel mower',
                    'snow blower', 'tiller', 'cultivator', 'chipper', 'shredder',
                    'pole saw', 'pruner', 'brush cutter', 'clearing saw'
                ],
                'keywords': [
                    'outdoor', 'yard work', 'landscaping', 'lawn care', 'tree work',
                    'brush clearing', 'property maintenance', 'acreage', 'commercial grade'
                ]
            },
            'power_tools': {
                'brands': [
                    'dewalt', 'makita', 'milwaukee', 'ryobi', 'bosch', 'craftsman',
                    'porter cable', 'black decker', 'metabo', 'festool', 'ridgid',
                    'kobalt', 'hart', 'bauer', 'hercules', 'skil'
                ],
                'products': [
                    'drill', 'impact driver', 'hammer drill', 'rotary hammer',
                    'circular saw', 'miter saw', 'table saw', 'jigsaw', 'reciprocating saw',
                    'angle grinder', 'orbital sander', 'belt sander', 'router',
                    'planer', 'nail gun', 'brad nailer', 'framing nailer',
                    'multitool', 'oscillating tool', 'dremel', 'rotary tool'
                ],
                'keywords': [
                    'cordless', 'brushless', 'battery', 'volt', 'amp hour', 'ah',
                    'torque', 'rpm', 'construction', 'woodworking', 'metalworking',
                    'professional grade', 'contractor', 'trade'
                ]
            },
            'appliances': {
                'brands': [
                    'whirlpool', 'ge', 'samsung', 'lg', 'frigidaire', 'kenmore',
                    'maytag', 'electrolux', 'bosch', 'miele', 'speed queen'
                ],
                'products': [
                    # Major appliances (non-cooking)
                    'washing machine', 'washer', 'dryer', 'clothes dryer',
                    'refrigerator', 'fridge', 'freezer', 'chest freezer',
                    'dishwasher', 'garbage disposal', 'water heater',
                    'hvac', 'furnace', 'air conditioner', 'ac unit', 'heat pump',
                    
                    # Small appliances (utility focused)
                    'vacuum', 'shop vac', 'wet dry vac', 'carpet cleaner',
                    'air purifier', 'dehumidifier', 'humidifier', 'space heater',
                    'window fan', 'exhaust fan', 'bathroom fan'
                ],
                'keywords': [
                    'energy efficient', 'star rated', 'capacity', 'cubic feet',
                    'front load', 'top load', 'stackable', 'compact',
                    'commercial grade', 'heavy duty', 'reliability'
                ]
            }
        }
        
        # Enhanced demand signal patterns
        self.demand_patterns = {
            'buying_intent': {
                'patterns': [
                    r'(?i)(what\'s the best|which.*should i|recommend|suggestions for)',
                    r'(?i)(looking for|shopping for|need to buy|buying)',
                    r'(?i)(budget.*for|price range|how much|worth it)',
                    r'(?i)(upgrade from|replace.*with|better than)'
                ],
                'weight': 3.0
            },
            'troubleshooting': {
                'patterns': [
                    r'(?i)(not working|broken|problem with|issue with)',
                    r'(?i)(won\'t start|died|failed|stopped working)',
                    r'(?i)(repair|fix|troubleshoot|diagnose)',
                    r'(?i)(error|fault|malfunction|acting up)'
                ],
                'weight': 2.5
            },
            'learning_intent': {
                'patterns': [
                    r'(?i)(how to|how do i|tutorial|guide|instructions)',
                    r'(?i)(new to|beginner|first time|getting started)',
                    r'(?i)(learn|teach me|explain|show me)',
                    r'(?i)(tips|advice|best practices)'
                ],
                'weight': 2.0
            },
            'comparison': {
                'patterns': [
                    r'(?i)(vs|versus|compare|comparison|difference)',
                    r'(?i)(better|superior|pros and cons)',
                    r'(?i)(alternative|substitute|instead of)'
                ],
                'weight': 1.8
            },
            'maintenance': {
                'patterns': [
                    r'(?i)(maintain|maintenance|care|service)',
                    r'(?i)(clean|oil|sharpen|tune up)',
                    r'(?i)(extend life|last longer|keep running)'
                ],
                'weight': 1.5
            }
        }

    def calculate_relevance_score(self, post_text, title_text):
        """Calculate relevance score for your specific niches"""
        combined_text = (title_text + " " + post_text).lower()
        score = 0
        matched_elements = {
            'brands': [],
            'products': [],
            'keywords': [],
            'intents': []
        }
        
        # Score by niche category
        for niche, data in self.niche_categories.items():
            niche_score = 0
            
            # Brand mentions (high value)
            for brand in data['brands']:
                if brand in combined_text:
                    niche_score += 5
                    matched_elements['brands'].append(brand)
            
            # Product mentions (high value)
            for product in data['products']:
                if product in combined_text:
                    niche_score += 4
                    matched_elements['products'].append(product)
            
            # Niche keywords (medium value)
            for keyword in data['keywords']:
                if keyword in combined_text:
                    niche_score += 2
                    matched_elements['keywords'].append(keyword)
            
            score += niche_score
        
        # Intent scoring
        for intent, config in self.demand_patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, combined_text):
                    score += config['weight']
                    matched_elements['intents'].append(intent)
        
        # Quality multipliers
        if len(combined_text.split()) > 20:  # Substantial posts
            score *= 1.2
        
        if any(word in combined_text for word in ['professional', 'commercial', 'contractor']):
            score *= 1.3  # Professional use cases are valuable
        
        # Penalty for food/cooking content you don't want
        cooking_words = ['recipe', 'cooking', 'baking', 'food', 'kitchen', 'meal', 'chef']
        if any(word in combined_text for word in cooking_words):
            score *= 0.3  # Heavy penalty
        
        return score, matched_elements

    def extract_posts(self, subreddit_name, time_filter='week', limit=100):
        """Extract posts with improved relevance filtering"""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            
            print(f"Scanning r/{subreddit_name}...")
            
            for submission in subreddit.top(time_filter=time_filter, limit=limit):
                # Quick pre-filter
                text_content = (submission.title + " " + submission.selftext).lower()
                
                # Check if it mentions any of your niche terms
                has_niche_content = False
                for niche_data in self.niche_categories.values():
                    if (any(brand in text_content for brand in niche_data['brands']) or
                        any(product in text_content for product in niche_data['products'])):
                        has_niche_content = True
                        break
                
                if not has_niche_content:
                    continue
                
                # Calculate detailed relevance
                relevance_score, matched_elements = self.calculate_relevance_score(
                    submission.selftext, submission.title
                )
                
                # Only include posts with meaningful relevance
                if relevance_score >= 3.0:
                    posts.append({
                        'subreddit': subreddit_name,
                        'title': submission.title,
                        'selftext': submission.selftext[:1000],  # Limit length
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'created_utc': submission.created_utc,
                        'url': submission.url,
                        'permalink': f"https://reddit.com{submission.permalink}",
                        'relevance_score': relevance_score,
                        'matched_brands': matched_elements['brands'],
                        'matched_products': matched_elements['products'],
                        'matched_keywords': matched_elements['keywords'],
                        'detected_intents': matched_elements['intents']
                    })
            
            time.sleep(1)
            return posts
            
        except Exception as e:
            print(f"Error processing r/{subreddit_name}: {e}")
            return []

    def analyze_demand_themes(self, posts):
        """Smart categorization based on your niches"""
        themes = defaultdict(list)
        
        for post in posts:
            content = (post['title'] + " " + post['selftext']).lower()
            categorized = False
            
            # Outdoor Power Equipment
            ope_indicators = ['chainsaw', 'leaf blower', 'trimmer', 'mower', 'pressure washer', 'snow blower']
            if any(indicator in content for indicator in ope_indicators):
                themes['outdoor_power_equipment'].append(post)
                categorized = True
            
            # Power Tools
            elif any(tool in content for tool in ['drill', 'saw', 'grinder', 'nailer', 'driver', 'router']):
                themes['power_tools'].append(post)
                categorized = True
            
            # Major Appliances
            elif any(app in content for app in ['washer', 'dryer', 'refrigerator', 'dishwasher', 'hvac', 'furnace']):
                themes['major_appliances'].append(post)
                categorized = True
            
            # Small Appliances (utility focus)
            elif any(app in content for app in ['vacuum', 'air purifier', 'dehumidifier', 'space heater']):
                themes['utility_appliances'].append(post)
                categorized = True
            
            # Intent-based sub-categories
            if categorized:
                intents = post.get('detected_intents', [])
                if 'buying_intent' in intents:
                    themes['buying_advice'].append(post)
                if 'troubleshooting' in intents:
                    themes['repair_help'].append(post)
                if 'learning_intent' in intents:
                    themes['how_to_guides'].append(post)
                if 'comparison' in intents:
                    themes['product_comparisons'].append(post)
        
        return dict(themes)

    def generate_content_opportunities(self, themes):
        """Generate content opportunities focused on your niches"""
        opportunities = []
        
        theme_mappings = {
            'outdoor_power_equipment': {
                'content_angle': 'OPE Reviews, Maintenance & Buying Guides',
                'example_videos': [
                    'Chainsaw buying guide 2024', 
                    'Pressure washer comparison', 
                    'Lawn mower maintenance tips'
                ],
                'hashtags': ['#chainsaw', '#lawnmower', '#pressurewasher', '#OPE', '#yardwork']
            },
            'power_tools': {
                'content_angle': 'Power Tool Reviews & Professional Tips',
                'example_videos': [
                    'Best cordless drills 2024', 
                    'Miter saw buying guide', 
                    'Tool storage solutions'
                ],
                'hashtags': ['#powertools', '#cordlesstools', '#dewalt', '#milwaukee', '#contractor']
            },
            'major_appliances': {
                'content_angle': 'Appliance Buying Guides & Maintenance',
                'example_videos': [
                    'Washer and dryer buying guide', 
                    'HVAC maintenance tips', 
                    'Refrigerator troubleshooting'
                ],
                'hashtags': ['#appliances', '#HVAC', '#homeappliances', '#maintenance']
            },
            'utility_appliances': {
                'content_angle': 'Home Utility Equipment Reviews',
                'example_videos': [
                    'Best shop vacuums 2024', 
                    'Air purifier comparison', 
                    'Dehumidifier buying guide'
                ],
                'hashtags': ['#shopvac', '#airpurifier', '#homeimprovement', '#utility']
            },
            'buying_advice': {
                'content_angle': 'Product Buying Guides & Recommendations',
                'example_videos': [
                    'What to look for when buying...', 
                    'Best value tools under $200', 
                    'Professional vs homeowner grade'
                ],
                'hashtags': ['#buyingguide', '#recommendations', '#bestvalue', '#professional']
            },
            'repair_help': {
                'content_angle': 'Troubleshooting & Repair Tutorials',
                'example_videos': [
                    'Common tool problems and fixes', 
                    'Appliance troubleshooting guide', 
                    'When to repair vs replace'
                ],
                'hashtags': ['#repair', '#troubleshooting', '#DIYrepair', '#maintenance']
            }
        }
        
        for theme, posts in themes.items():
            if len(posts) >= 2:  # Lower threshold since content is more targeted
                avg_score = sum(p['score'] for p in posts) / len(posts)
                avg_relevance = sum(p.get('relevance_score', 0) for p in posts) / len(posts)
                total_comments = sum(p['num_comments'] for p in posts)
                
                opportunities.append({
                    'theme': theme,
                    'post_count': len(posts),
                    'avg_engagement': avg_score,
                    'avg_relevance': avg_relevance,
                    'total_comments': total_comments,
                    'demand_strength': len(posts) * avg_score * avg_relevance,
                    'sample_posts': sorted(posts, key=lambda x: x.get('relevance_score', 0), reverse=True)[:3],
                    **theme_mappings.get(theme, {
                        'content_angle': f'{theme.replace("_", " ").title()} Content',
                        'example_videos': ['Custom content needed'],
                        'hashtags': [f'#{theme}']
                    })
                })
        
        opportunities.sort(key=lambda x: x['demand_strength'], reverse=True)
        return opportunities

    def run_full_analysis(self):
        """Run the improved analysis focused on your niches"""
        print("Starting Improved Reddit Demand Analysis...")
        print(f"Focused on: Outdoor Power Equipment, Power Tools, Appliances")
        print(f"Target subreddits: {len(self.target_subreddits)}")
        
        all_posts = []
        for subreddit in self.target_subreddits:
            posts = self.extract_posts(subreddit, limit=50)  # Smaller but more targeted
            all_posts.extend(posts)
            time.sleep(2)
        
        print(f"Total relevant posts collected: {len(all_posts)}")
        if not all_posts:
            print("No relevant posts found matching your niches")
            return
        
        # Show relevance distribution
        relevance_scores = [p.get('relevance_score', 0) for p in all_posts]
        print(f"Relevance scores - Avg: {np.mean(relevance_scores):.1f}, Max: {max(relevance_scores):.1f}")
        
        themes = self.analyze_demand_themes(all_posts)
        print(f"Demand themes identified: {len(themes)}")
        
        opportunities = self.generate_content_opportunities(themes)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("reports", exist_ok=True)
        
        latest_json_path = os.path.join("reports", "latest.json")
        with open(latest_json_path, "w") as f:
            json.dump(opportunities, f, indent=2, default=str)
        
        archive_path = f'reports/improved_analysis_{timestamp}.json'
        with open(archive_path, 'w') as f:
            json.dump(opportunities, f, indent=2, default=str)
        
        self.create_summary_report(opportunities, timestamp)
        print(f"✅ Improved analysis complete! Reports saved with timestamp: {timestamp}")
        return opportunities

    def create_summary_report(self, opportunities, timestamp):
        """Create summary with niche focus"""
        lines = [f"# Improved Reddit Demand Analysis ({timestamp})", ""]
        lines.append(f"**Focus Areas:** Outdoor Power Equipment, Power Tools, Appliances")
        lines.append("")
        
        for opp in opportunities[:10]:
            lines.append(f"## {opp['theme'].replace('_', ' ').title()}")
            lines.append(f"- **Content Angle:** {opp['content_angle']}")
            lines.append(f"- **Posts:** {opp['post_count']} | **Avg Relevance:** {opp.get('avg_relevance', 0):.1f}")
            lines.append(f"- **Engagement:** {opp['avg_engagement']:.1f} | **Comments:** {opp['total_comments']}")
            lines.append("### Top Posts:")
            
            for post in opp['sample_posts']:
                brand_tags = ', '.join(post.get('matched_brands', [])[:3])
                product_tags = ', '.join(post.get('matched_products', [])[:3])
                lines.append(f"- [{post['title']}]({post['permalink']}) ({post['score']} pts)")
                if brand_tags:
                    lines.append(f"  - Brands: {brand_tags}")
                if product_tags:
                    lines.append(f"  - Products: {product_tags}")
            
            lines.append(f"**Hashtags:** {', '.join(opp['hashtags'])}")
            lines.append("")
        
        report_path = os.path.join("reports", f"improved_summary_{timestamp}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

if __name__ == "__main__":
    mapper = ImprovedRedditDemandMapper()
    mapper.run_full_analysis()