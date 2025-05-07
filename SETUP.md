# Setup Instructions for App Reviews AI

This document provides detailed setup instructions for getting the App Reviews AI system up and running.

## Dependency Issues Resolution

If you're encountering dependency conflicts between ChromaDB and OpenAI, follow these steps to resolve them:

### Version Compatibility

There's a known issue with newer versions of OpenAI and ChromaDB. The solution is to use versions that are compatible with each other:

1. First, create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the correct versions of dependencies:
```bash
pip install openai==0.28.1
pip install chromadb==0.4.18
```

3. Install the rest of the dependencies:
```bash
pip install -r requirements.txt
```

### OpenAI API Key Setup

For the LLM functionality to work, you need to set up your OpenAI API key:

1. Get your API key from [OpenAI](https://platform.openai.com/api-keys)
2. Create a `.env` file in the project root:
```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

3. Alternatively, set it as an environment variable:
```bash
export OPENAI_API_KEY=your_key_here  # On Windows: set OPENAI_API_KEY=your_key_here
```

## Running the System with Mock Data

If you don't want to use the real Google Play Store API or want to demo the system without internet access:

1. The system is configured to use mock data by default
2. Mock data is realistic and representative of real app reviews
3. To run with mock data:
```bash
python run.py --max-reviews 50
```

## Running with Real Data

If you want to fetch real data from the Google Play Store:

1. Update the `use_mock` parameter in `src/modules/acquisition/google_play.py` to `False`
2. Run the system with a specific app ID:
```bash
python run.py --app-id com.example.app --max-reviews 50
```

## Troubleshooting

### ChromaDB "proxies" error

Error: `Client.__init__() got an unexpected keyword argument 'proxies'`

Resolution:
- Use OpenAI 0.28.1 and ChromaDB 0.4.18
- Ensure you're using the correct model name: "text-embedding-ada-002"

### OpenAI API errors

Error: `Error generating text with OpenAI: You didn't provide an API key...`

Resolution:
- Check that your API key is set correctly in the `.env` file or as an environment variable
- Verify the API key is valid and has sufficient credits

### Empty Output

Issue: System runs but produces empty charts or minimal output

Resolution:
- Increase the number of mock reviews with `--max-reviews 100`
- Check the output in the reports directory to see the generated visualizations