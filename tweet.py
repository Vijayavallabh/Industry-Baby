from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time

def scrape_tweets(username, tweet_limit=10):
    # Initialize the WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode
    options.add_argument('--disable-gpu')  # Disable GPU rendering
    options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe" 
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        # Open Twitter and navigate to the user's profile
        url = f"https://twitter.com/{username}"
        driver.get(url)
        time.sleep(5)  # Wait for the page to load
        
        # Scroll and collect tweets
        tweets = []
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        while len(tweets) < tweet_limit:
            # Find tweets on the page
            tweet_elements = driver.find_elements(By.XPATH, "//article[@data-testid='tweet']//div[@lang]")
            for tweet_element in tweet_elements:
                tweet_text = tweet_element.text
                if tweet_text not in tweets:
                    tweets.append(tweet_text)
                    if len(tweets) >= tweet_limit:
                        break
            
            # Scroll down
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)  # Wait for new tweets to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break  # Break if no new tweets are loaded
            last_height = new_height
        
        # Print the collected tweets
        for idx, tweet in enumerate(tweets, start=1):
            print(f"{idx}: {tweet}")
        
        return tweets
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the driver
        driver.quit()

# Usage
if __name__ == "__main__":
    username = input("Enter the Twitter username: ").strip()
    tweet_limit = int(input("Enter the number of tweets to fetch: "))
    tweets = scrape_tweets(username, tweet_limit)
    print(f"\nFetched {len(tweets)} tweets for user @{username}")
