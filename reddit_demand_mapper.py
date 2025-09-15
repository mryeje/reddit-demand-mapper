import praw
import pandas as pd
import json
import os
import re
from datetime import datetime
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
            'Repair', 'AskEngineers', 'whatisthisthing',
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

    def extract_posts(self, subreddit_name, time_filter='week', limit=200):
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
                        post_data = {
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
                        }
                        posts.append(post_data)
            
            time.sleep(1)
            return posts
            
        except Exception as e:
            print(f"Error processing r/{subreddit_name}: {e}")
            return []

    def extract_comments(self, submission_url, max_comments=20):
        """Extract high-engagement comments from a submission"""
        try:
            submission = self.reddit.submission(url=submission_url)
            submission.comments.replace_more(limit=0)
            
            valuable_comments = []
            for comment in submission.comments[:max_comments]:
                if comment.score >= 3:
                    comment_text = comment.body.lower()
                    demand_signals = []
                    for pattern in self.demand_patterns:
                        matches = re.findall(pattern, comment_text)
                        demand_signals.extend(matches)
                    
                    if demand_signals:
                        valuable_comments.append({
                            'comment_body': comment.body[:300],
                            'score': comment.score,
                            'demand_signals': demand_signals
                        })
            
            return valuable_comments
            
        except Exception as e:
            print(f"Error extracting comments: {e}")
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
                
                opportunity = {
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
                }
                opportunities.append(opportunity)
        
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
        
        # Save all reports in reports/
        pd.DataFrame(all_posts).to_csv(f'reports/reddit_posts_{timestamp}.csv', index=False)
        with open(f'reports/tiktok_opportunities_{timestamp}.json', 'w') as f:
            json.dump(opportunities, f, indent=2, default=str)
        self.create_summary_report(opportunities, timestamp)
        
        print(f"Analysis complete! Reports saved in 'reports/' with timestamp: {timestamp}")
        return opportunities

    def create_summary_report(self, opportunities, timestamp):
        report = f"""
# TikTok Content Opportunities Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Top Opportunities (Ranked by Demand Strength)

"""
        for i, opp in enumerate(opportunities[:10], 1):
            report += f"""
### {i}. {opp['content_angle']}
- **Demand Strength**: {opp['demand_strength']:.1f}
- **Posts Found**: {opp['post_count']}
- **Average Engagement**: {opp['avg_engagement']:.1f}
- **Total Comments**: {opp['total_comments']}

**Example TikTok Videos:**
{chr(10).join(f"- {video}" for video in opp['example_videos'])}

**Suggested Hashtags:**
{' '.join(opp['hashtags'])}

**Sample Reddit Posts:**
"""
            for post in opp['sample_posts'][:2]:
                report += f"""
- **r/{post['subreddit']}**: "{post['title']}" ({post['score']} upvotes, {post['num_comments']} comments)
  URL: {post['permalink']}
"""
        
        with open(f'reports/opportunities_summary_{timestamp}.md', 'w') as f:
            f.write(report)

if __name__ == "__main__":
    mapper = RedditDemandMapper()
    opportunities = mapper.run_full_analysis()
    
    if opportunities:
        print(f"\nTop 3 TikTok Opportunities:")
        for i, opp in enumerate(opportunities[:3], 1):
            print(f"{i}. {opp['content_angle']} (Demand: {opp['demand_strength']:.1f})")
