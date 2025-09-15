import praw
import pandas as pd
import json
import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
import time

class RedditDemandMapper:
    def __init__(self):
        # Initialize Reddit API connection
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent='DemandMapper/1.0'
        )
        
        # Target subreddits for OPE, power tools, and appliances
        self.target_subreddits = [
            'Tools', 'Dewalt', 'Makita', 'Milwaukee', 'Ryobi',
            'woodworking', 'Carpentry', 'HomeImprovement', 'DIY',
            'electricians', 'Plumbing', 'landscaping', 'lawncare',
            'powerwashing', 'AutoDetailing', 'MechanicAdvice',
            'appliances', 'BuyItForLife', 'fixit', 'HomeAppliances',
            'Appliances', 'appliancerepair', 'Frugal', 'BudgetFood',
            'MealPrepSunday', 'cookingforbeginners', 'Baking',
            'fixit', 'Repair', 'AskEngineers', 'whatisthisthing',
            'HelpMeFind', 'NoStupidQuestions', 'explainlikeimfive'
        ]
        
        # Demand signal patterns
        self.demand_patterns = [
            r"(?i)(how do i|how to|can someone|does anyone know|help with|tutorial)",
            r"(?i)(what's the best|which tool|recommend|suggestions for|advice)",
            r"(?i)(struggling with|can't figure out|having trouble|need help)",
            r"(?i)(wish someone would|why isn't there|looking for a video|tutorial)",
            r"(?i)(keeps breaking|not working|failed|died|stopped)",
            r"(?i)(expensive|budget|cheap|alternative|dupe)",
            r"(?i)(beginner|new to|first time|don't know)",
            r"(?i)(apartment|rental|small space|limited space)"
        ]
        
        # Keywords for OPE/Tools/Appliances
        self.niche_keywords = [
            'chainsaw', 'leaf blower', 'hedge trimmer', 'weed eater', 'string trimmer',
            'pressure washer', 'lawn mower', 'zero turn', 'riding mower', 'push mower',
            'generator', 'air compressor', 'wood chipper', 'tiller',
            'drill', 'impact driver', 'circular saw', 'miter saw', 'table saw',
            'jigsaw', 'reciprocating saw', 'angle grinder', 'router', 'planer',
            'nail gun', 'stapler', 'multitool', 'oscillating', 'battery',
            'cordless', 'brushless', 'torque', 'chuck', 'blade',
            'refrigerator', 'dishwasher', 'washing machine', 'dryer', 'oven',
            'microwave', 'air fryer', 'instant pot', 'food processor', 'blender',
            'mixer', 'toaster', 'coffee maker', 'vacuum', 'air purifier'
        ]

    def extract_posts(self, subreddit_name, time_filter='week', limit=50):
        """Extract posts from a subreddit that match demand patterns"""
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = []
            
            print(f"Scanning r/{subreddit_name}...")
            
            for submission in subreddit.top(time_filter=time_filter, limit=limit):
                text_content = (submission.title + " " + submission.selftext).lower()
                
                if any(keyword in text_content for keyword in self.niche_keywords):
                    demand_signals = []
                    for pattern in self.demand_patterns:
                        matches = re.findall(pattern, text_content)
                        demand_signals.extend(matches)
                    
                    if demand_signals or submission.num_comments >= 5:
                        posts.append({
                            'subreddit': subreddit_name,
                            'title': submission.title,
                            'selftext': submission.selftext[:500],
                            'score': submission.score,
                            'num_comments': submission.num_comments,
                            'created_utc': submission.created_utc,
                            'url': submission.url,
                            'permalink': f"https://reddit.com{submission.permalink}",
                            'demand_signals': demand_signals,
                            'matched_keywords': [kw for kw in self.niche_keywords if kw in text_content]
                        })
            
            time.sleep(1)
            return posts
        except Exception as e:
            print(f"Error processing r/{subreddit_name}: {e}")
            return []

    def analyze_demand_themes(self, posts):
        """Analyze posts to identify recurring demand themes"""
        themes = defaultdict(list)
        for post in posts:
            content = (post['title'] + " " + post['selftext']).lower()
            if any(word in content for word in ['chainsaw', 'saw', 'cutting']):
                themes['cutting_tools'].append(post)
            elif any(word in content for word in ['drill', 'impact', 'driver']):
                themes['drilling_tools'].append(post)
            elif any(word in content for word in ['lawn', 'grass', 'mower']):
                themes['lawn_care'].append(post)
            elif any(word in content for word in ['pressure', 'wash', 'clean']):
                themes['cleaning_equipment'].append(post)
            elif any(word in content for word in ['kitchen', 'cook', 'appliance']):
                themes['kitchen_appliances'].append(post)
            elif any(word in content for word in ['repair', 'fix', 'broken']):
                themes['repair_help'].append(post)
            elif any(word in content for word in ['beginner', 'first', 'new']):
                themes['beginner_guidance'].append(post)
            elif any(word in content for word in ['budget', 'cheap', 'affordable']):
                themes['budget_options'].append(post)
            else:
                themes['general'].append(post)
        return dict(themes)

    def generate_tiktok_opportunities(self, themes):
        """Generate TikTok content opportunities from demand themes"""
        opportunities = []
        theme_mappings = {
            'cutting_tools': {
                'content_angle': 'Chainsaw & Cutting Tool Reviews/Tips',
                'example_videos': ['Chainsaw maintenance in 60 seconds', 'Best budget chainsaws 2024'],
                'hashtags': ['#chainsaw', '#powertools', '#DIY', '#woodworking']
            },
            'drilling_tools': {
                'content_angle': 'Drill & Driver Comparisons/Tutorials',
                'example_videos': ['Impact driver vs regular drill', 'Drill bit guide for beginners'],
                'hashtags': ['#drill', '#impactdriver', '#powertools', '#construction']
            },
            'lawn_care': {
                'content_angle': 'Lawn Equipment Reviews & Tips',
                'example_videos': ['Mower buying guide', 'Lawn care mistakes to avoid'],
                'hashtags': ['#lawncare', '#mower', '#landscaping', '#yardwork']
            },
            'cleaning_equipment': {
                'content_angle': 'Pressure Washing & Cleaning Equipment',
                'example_videos': ['Pressure washer setup guide', 'Before/after cleaning reveals'],
                'hashtags': ['#pressurewashing', '#cleaning', '#satisfying', '#powerwashing']
            },
            'kitchen_appliances': {
                'content_angle': 'Appliance Reviews & Kitchen Hacks',
                'example_videos': ['Air fryer vs oven comparison', 'Kitchen appliance buying guide'],
                'hashtags': ['#kitchenappliances', '#airfryer', '#kitchenhacks', '#cooking']
            },
            'repair_help': {
                'content_angle': 'DIY Repair Tutorials',
                'example_videos': ['Fix it yourself tutorials', 'Common appliance problems'],
                'hashtags': ['#DIYrepair', '#fixit', '#repair', '#maintenance']
            },
            'beginner_guidance': {
                'content_angle': 'Tool & Equipment Education for Beginners',
                'example_videos': ['Tools every homeowner needs', 'Beginner tool buying guide'],
                'hashtags': ['#beginnertools', '#DIYbeginner', '#homeimprovement', '#tools101']
            },
            'budget_options': {
                'content_angle': 'Budget Tool & Appliance Reviews',
                'example_videos': ['Best budget power tools', 'Cheap vs expensive tool comparison'],
                'hashtags': ['#budgettools', '#affordabletools', '#tooldeals', '#budgetDIY']
            }
        }
        for theme, posts in themes.items():
            if len(posts) >= 3:
                avg_score = sum(p['score'] for p in posts) / len(posts)
                total_comments = sum(p['num_comments'] for p in posts)
                opportunities.append({
                    'theme': theme,
                    'post_count': len(posts),
                    'avg_engagement': avg_score,
                    'total_comments': total_comments,
                    'demand_strength': len(posts) * avg_score,
                    'sample_posts': posts[:3],
                    **theme_mappings.get(theme, {
                        'content_angle': f'{theme.replace("_", " ").title()} Content',
                        'example_videos': ['Custom content needed'],
                        'hashtags': [f'#{theme}']
                    })
                })
        opportunities.sort(key=lambda x: x['demand_strength'], reverse=True)
        return opportunities

    def run_full_analysis(self):
        print("Starting Reddit Demand Mapping Analysis...")
        print(f"Target subreddits: {len(self.target_subreddits)}")
        
        all_posts = []
        for subreddit in self.target_subreddits:
            posts = self.extract_posts(subreddit)
            all_posts.extend(posts)
            time.sleep(2)
        
        print(f"Total posts collected: {len(all_posts)}")
        if not all_posts:
            print("No posts found matching criteria")
            return
        
        themes = self.analyze_demand_themes(all_posts)
        print(f"Demand themes identified: {len(themes)}")
        
        opportunities = self.generate_tiktok_opportunities(themes)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save latest.json for dashboard
        latest_json_path = os.path.join("reports", "latest.json")
        os.makedirs("reports", exist_ok=True)
        with open(latest_json_path, "w") as f:
            json.dump(opportunities, f, indent=2, default=str)

        # Save historical reports
        pd.DataFrame(all_posts).to_csv(f'reports/reddit_posts_{timestamp}.csv', index=False)
        with open(f'reports/tiktok_opportunities_{timestamp}.json', 'w') as f:
            json.dump(opportunities, f, indent=2, default=str)

        self.create_summary_report(opportunities, timestamp)
        print(f"Analysis complete! Reports saved in 'reports/' with timestamp: {timestamp}")
        return opportunities

    def create_summary_report(self, opportunities, timestamp):
        """Create a simple markdown summary report"""
        lines = [f"# Reddit Demand Mapping Report ({timestamp})", ""]
        top_themes = opportunities[:10]
        for opp in top_themes:
            lines.append(f"## {opp['theme'].replace('_', ' ').title()}")
            lines.append(f"Content Angle: {opp['content_angle']}")
            lines.append(f"Average Engagement: {opp['avg_engagement']:.2f}")
            lines.append(f"Total Comments: {opp['total_comments']}")
            lines.append(f"Sample Posts:")
            for post in opp['sample_posts']:
                lines.append(f"- [{post['title']}]({post['permalink']}) ({post['score']} pts, {post['num_comments']} comments)")
            lines.append(f"Suggested Hashtags: {', '.join(opp['hashtags'])}")
            lines.append("")
        report_path = os.path.join("reports", f"opportunities_summary_{timestamp}.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

if __name__ == "__main__":
    mapper = RedditDemandMapper()
    mapper.run_full_analysis()
