"""X (Twitter) MCP server — read-only, cookie-based auth via twikit."""

import os
import sys
from mcp.server.fastmcp import FastMCP

# Ensure stdout/stderr can handle non-ASCII characters (emojis, etc.)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
from twikit import Client

mcp = FastMCP("x")

_client: Client | None = None

COOKIE_PATH = os.getenv("X_COOKIE_PATH", "/tmp/x_cookies.json")


async def get_client() -> Client:
    global _client
    if _client is not None:
        return _client

    if not os.path.exists(COOKIE_PATH):
        raise RuntimeError(
            f"No X cookie file found at {COOKIE_PATH}. "
            "Run setup first: uv run python x-mcp/x-mcp-server/setup.py"
        )

    client = Client("en-US")
    client.load_cookies(COOKIE_PATH)
    _client = client
    return client


def _format_tweet(tweet) -> dict:
    return {
        "id": tweet.id,
        "text": tweet.text,
        "author": tweet.user.screen_name,
        "created_at": str(tweet.created_at),
        "likes": tweet.favorite_count,
        "retweets": tweet.retweet_count,
        "url": f"https://x.com/{tweet.user.screen_name}/status/{tweet.id}",
    }


@mcp.tool()
async def get_user_tweets(username: str, count: int = 20) -> list[dict]:
    """Get recent tweets from a public X/Twitter account.

    Args:
        username: The X/Twitter username (without @), e.g. "elonmusk"
        count: Number of tweets to fetch (max 40)
    """
    client = await get_client()
    user = await client.get_user_by_screen_name(username)
    tweets = await client.get_user_tweets(user.id, "Tweets", count=min(count, 40))
    return [_format_tweet(t) for t in tweets]


@mcp.tool()
async def search_tweets(query: str, count: int = 20) -> list[dict]:
    """Search X/Twitter for tweets matching a query.

    Args:
        query: Search query — supports keywords, hashtags, or 'from:username' syntax
        count: Number of results to return (max 40)
    """
    client = await get_client()
    tweets = await client.search_tweet(query, "Latest", count=min(count, 40))
    return [_format_tweet(t) for t in tweets]


if __name__ == "__main__":
    mcp.run()
