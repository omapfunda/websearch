from flask import Flask, render_template, request, jsonify
import requests
from mistralai import Mistral
import os
from dotenv import load_dotenv
import json
from tavily import TavilyClient
import time

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Validate required environment variables are present
required_env_vars = ['MISTRAL_API_KEY', 'TAVILY_API_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize clients with API keys from environment variables
mistral_client = Mistral(api_key=os.getenv('MISTRAL_API_KEY'))
tavily_client = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

def extract_content(url):
    """
    Extract full content from a URL using Tavily Extract API
    
    Args:
        url (str): URL to extract content from
        
    Returns:
        str: Extracted content or None if extraction fails
    """
    try:
        response = tavily_client.extract_content(url=url)
        return response.get('content')
    except Exception as e:
        app.logger.error(f"Content extraction error for URL {url}: {str(e)}")
        return None

def search_tavily(query, max_results=5):
    """
    Perform a Tavily web search and extract full content from results
    
    Args:
        query (str): Search query
        max_results (int): Maximum number of results to return
        
    Returns:
        dict: Formatted search results with full content or None if search fails
    """
    try:
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced"
        )
        
        # Transform and validate Tavily results, including full content extraction
        results = []
        for result in response.get('results', []):
            if all(key in result for key in ['title', 'url', 'content']):
                # Extract full content for each result
                full_content = extract_content(result['url'])
                # Add a delay to respect rate limits
                time.sleep(1)
                
                results.append({
                    'name': result['title'],
                    'url': result['url'],
                    'snippet': result['content'],
                    'full_content': full_content or result['content']  # Fallback to snippet if extraction fails
                })
        
        return {'webPages': {'value': results}} if results else None
    
    except Exception as e:
        app.logger.error(f"Tavily search error: {str(e)}")
        return None

def process_with_mistral(search_results, user_query):
    """
    Process search results with Mistral LLM using full content when available
    
    Args:
        search_results (dict): Search results from Tavily
        user_query (str): Original user query
        
    Returns:
        str: Processed response or None if processing fails
    """
    try:
        # Input validation
        if not isinstance(search_results, dict) or 'webPages' not in search_results:
            raise ValueError("Invalid search results format")
        
        # Prepare context from search results, using full content
        context_parts = ["Based on the following sources:\n"]
        for idx, result in enumerate(search_results['webPages']['value'], 1):
            context_parts.append(f"\nSource {idx} ({result['url']}):\n{result['full_content']}")
        
        prompt = (
            f"{' '.join(context_parts)}\n\n"
            f"User Query: {user_query}\n\n"
            "Please provide a comprehensive answer based on these sources, synthesizing the full content provided:"
        )
        
        response = mistral_client.chat.complete(
            model="mistral-medium",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000  # Increased token limit to handle longer content
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        app.logger.error(f"Mistral processing error: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({"error": "Empty query"}), 400
        
        # Perform search with content extraction
        search_results = search_tavily(query)
        if not search_results:
            return jsonify({"error": "Search failed"}), 500
        
        # Process results using full content
        processed_response = process_with_mistral(search_results, query)
        if not processed_response:
            return jsonify({"error": "Processing failed"}), 500
        
        return jsonify({
            "search_results": search_results['webPages']['value'],
            "processed_response": processed_response
        })
    
    except Exception as e:
        app.logger.error(f"Search endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(port=5001)