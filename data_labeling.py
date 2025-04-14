import re
import pandas as pd
from openai import OpenAI
import json
import time
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

# Set OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_cleaned_data(file_path):
    """Load the cleaned conversation dataset"""
    print(f"Loading cleaned data from {file_path}...")
    return pd.read_csv(file_path)

def create_batch_prompt(conversations_batch):
    """Create a prompt for the OpenAI API to label multiple conversations in one request"""
    conversations_text = ""
    for idx, (conv_id, context, response) in enumerate(conversations_batch):
        conversations_text += f"\nCONVERSATION {idx+1}:\n"
        conversations_text += f"Patient's message: {context}\n"
        conversations_text += f"Therapist's response: {response}\n"
        conversations_text += f"---\n"
    
    prompt = f"""You are an expert mental health professional who is analyzing therapy conversations.
    
Below are multiple patient-therapist conversation pairs:
{conversations_text}

For EACH conversation, analyze it and provide the following:
1. Mental health issues (list ALL that apply, upto 5 based on highest relevance): depression, anxiety, stress, trauma, substance abuse, 
   eating disorder, relationship issues, grief, identity issues, self-esteem, anger management, 
   suicidal thoughts, OCD, bipolar, schizophrenia, ADHD, PTSD, sleep disorders, 
   burnout, family conflict, work stress, academic stress, social anxiety, panic attacks, phobias,
   loneliness, life transitions, perfectionism, procrastination, financial stress, sexual health,
   body image, chronic illness, bullying, harassment, discrimination, cultural issues, religious/spiritual concerns,
   or any other issues you identify (please specify)

2. Response type (list ALL that apply, upto 5 based on highest relevance): validation/empathy, direct advice, psychoeducation, 
   clarifying questions, coping strategies, referral/escalation, reflection, reframing,
   active listening, normalization, challenging thoughts, goal setting, homework assignment,
   crisis intervention, emotional support, symptom management, skills training

3. Therapeutic approach evident in response (list ALL that apply, upto 5 based on highest relevance): CBT, DBT, psychodynamic, 
   person-centered, solution-focused, motivational interviewing, mindfulness-based, 
   existential, humanistic, behavioral, cognitive, trauma-informed, strength-based,
   acceptance and commitment therapy (ACT), narrative therapy, gestalt, eclectic

For each category, don't limit yourself to just one label if multiple apply.

Provide your answer in JSON format with the following structure:
{{
  "results": [
    {{
      "conversation_id": 1,
      "issues": ["issue1", "issue2", "issue3", ...],
      "response_types": ["type1", "type2", ...],
      "therapeutic_approaches": ["approach1", "approach2", ...]
    }},
    {{
      "conversation_id": 2,
      "issues": ["issue1", "issue2", "issue3", ...],
      "response_types": ["type1", "type2", ...],
      "therapeutic_approaches": ["approach1", "approach2", ...]
    }},
    ...
  ]
}}

Ensure you provide labels for all {len(conversations_batch)} conversations.
"""
    return prompt

def label_conversations_batch_with_openai(conversations_batch, model="gpt-4o-mini"):
    """Use OpenAI API to label a batch of conversations in a single request"""
    prompt = create_batch_prompt(conversations_batch)
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes mental health conversations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent outputs
            max_tokens=4000   # Ensure enough tokens for the response
        )
        
        # Extract the response
        result = completion.choices[0].message.content
        
        # Parse the JSON
        def clean_json_response(text):
            # Remove triple backticks and optional "json" marker
            text = re.sub(r"```(?:json)?", "", text)
            text = text.replace("```", "")
            return text.strip()
        try:
            parsed_result = json.loads(clean_json_response(result))
            return parsed_result
        except json.JSONDecodeError:
            print(f"Error parsing JSON: {result[:500]}...")  # Print first 500 chars
            return None
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def process_data_in_batches(df, batch_size=5, model="gpt-4o-mini"):
    """Process all conversations in batches, with multiple conversations per API call"""
    print(f"Processing {len(df)} conversations with batch size {batch_size}...")
    
    all_results = []
    conversation_batches = []
    batch_conversation_ids = []
    
    # Create batches of conversations
    for idx, row in df.iterrows():
        conversation_batches.append((row['ConversationID'], row['Context'], row['Response']))
        batch_conversation_ids.append(row['ConversationID'])
        
        # When batch is full or on last item, process the batch
        if len(conversation_batches) == batch_size or idx == len(df) - 1:
            print(f"Processing batch of {len(conversation_batches)} conversations...")
            batch_result = label_conversations_batch_with_openai(conversation_batches, model)
            
            if batch_result and 'results' in batch_result:
                # Map the results back to actual conversation IDs
                for i, result in enumerate(batch_result['results']):
                    if i < len(batch_conversation_ids):  # Ensure we don't go out of bounds
                        result_row = {
                            'ConversationID': batch_conversation_ids[i],
                            'issues': json.dumps(result.get('issues', [])),
                            'response_types': json.dumps(result.get('response_types', [])),
                            'therapeutic_approaches': json.dumps(result.get('therapeutic_approaches', []))
                        }
                        all_results.append(result_row)
            
            # Reset batches
            conversation_batches = []
            batch_conversation_ids = []
            
            # Sleep to avoid rate limiting
            time.sleep(2)

    # Convert results to DataFrame
    final_df = pd.DataFrame(all_results)
    # save the final df to a csv file
    final_df.to_csv("data/labeled_conversations.csv", index=False)
    return final_df

def analyze_labels(labeled_df):
    """Analyze the generated labels and print statistics"""
    print("\nLabel Analysis:")
    
    # Convert string representations of lists to actual lists
    df_analysis = labeled_df.copy()
    df_analysis['issues'] = df_analysis['issues'].apply(json.loads)
    df_analysis['response_types'] = df_analysis['response_types'].apply(json.loads)
    df_analysis['therapeutic_approaches'] = df_analysis['therapeutic_approaches'].apply(json.loads)
    
    # Analyze issues
    all_issues = []
    for issues_list in df_analysis['issues']:
        all_issues.extend(issues_list)
    
    issue_counts = pd.Series(all_issues).value_counts()
    print("\nIssues Distribution:")
    print(issue_counts.head(20))  # Show top 20 issues
    
    # Analyze response types
    all_response_types = []
    for types_list in df_analysis['response_types']:
        all_response_types.extend(types_list)
    
    response_type_counts = pd.Series(all_response_types).value_counts()
    print("\nResponse Types Distribution:")
    print(response_type_counts)
    
    # Analyze therapeutic approaches
    all_approaches = []
    for approaches_list in df_analysis['therapeutic_approaches']:
        all_approaches.extend(approaches_list)
    
    approach_counts = pd.Series(all_approaches).value_counts()
    print("\nTherapeutic Approaches Distribution:")
    print(approach_counts)
    
    # Calculate average number of labels per conversation
    avg_issues = df_analysis['issues'].apply(len).mean()
    avg_response_types = df_analysis['response_types'].apply(len).mean()
    avg_approaches = df_analysis['therapeutic_approaches'].apply(len).mean()
    
    print(f"\nAverage labels per conversation:")
    print(f"Issues: {avg_issues:.2f}")
    print(f"Response Types: {avg_response_types:.2f}")
    print(f"Therapeutic Approaches: {avg_approaches:.2f}")
    
    return labeled_df

def merge_with_original_data(original_df, labeled_df):
    """Merge labeled data with original conversations"""
    print("\nMerging labels with original data...")
    
    # Merge dataframes on ConversationID
    merged_df = pd.merge(original_df, labeled_df, on='ConversationID')
    
    print(f"Final dataset shape: {merged_df.shape}")
    return merged_df

def main():
    # Set file paths
    input_file = "data/cleaned_conversations_final.csv"
    output_file = "data/labeled_conversations.csv"
    final_output = "data/mental_health_conversations_labeled.csv"
    
    # Load cleaned data
    df = load_cleaned_data(input_file)
    
    # For testing with a smaller sample (comment out for full dataset)
    # sample_size = 1
    # df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    
    # Process data with OpenAI - using batches of 5 conversations per API call
    # Using gpt-4o-mini for larger context window to handle multiple conversations
    labeled_df = process_data_in_batches(df, batch_size=5, model="gpt-4o-mini")
    
    # Save labeled data
    labeled_df.to_csv(output_file, index=False)
    print(f"Saved labeled data to {output_file}")
    
    # Analyze labels
    analyzed_df = analyze_labels(labeled_df)
    
    # Merge with original data
    final_df = merge_with_original_data(df, analyzed_df)
    
    # Save final dataset
    final_df.to_csv(final_output, index=False)
    print(f"Saved final labeled dataset to {final_output}")
    
if __name__ == "__main__":
    main()