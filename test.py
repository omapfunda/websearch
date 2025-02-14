from tavily import TavilyClient

tavily_client = TavilyClient(api_key="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")
response = tavily_client.search("BAsed on the latest sources, Who are the top 5 players in the helicopters lessors market ?")

print(response)
