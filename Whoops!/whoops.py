import torch
import json
import random
import pandas as pd
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import re
import ast

class WHOOPSSceneGraphGenerator:
    def __init__(self):
        self.processor = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load LLaVA model for scene graph generation"""
        try:
            print("Loading LLaVA model for global scene graph generation...")
            model_name = "llava-hf/llava-1.5-13b-hf"
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            print(f"Successfully loaded {model_name}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _clean_scene_graph_output(self, text):
        """Clean and extract JSON from model output"""
        text = text.strip()
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return text
    
    def generate_global_scene_graph(self, image):
        """Generate comprehensive global scene graph from image only"""
        try:
            prompt = """USER: <image> For the provided image, generate a comprehensive scene graph in JSON format that includes the following:

1. Objects that are present in the image with their confidence scores.
2. Object attributes in the image.
3. Object relationships: Spatial and semantic relationships between objects with confidence scores.

Don't be repetitive.

Focus on accuracy and avoid hallucination. Only include objects and relationships that are clearly visible in the image. Provide confidence scores between 0.0 and 1.0 for all objects, and relationships.

Scene Graph (JSON format only):

A:"""
            
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=250,
                    do_sample=True,
                    temperature=0.15,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            if "ASSISTANT:" in generated_text:
                scene_graph_text = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                scene_graph_text = generated_text.replace(prompt.replace("<image>", ""), "").strip()
            
            scene_graph_text = self._clean_scene_graph_output(scene_graph_text)
            
            return scene_graph_text if scene_graph_text else '{"objects": [], "relationships": []}'
            
        except Exception as e:
            return f'{{"error": "Error generating global scene graph: {str(e)}"}}'
    
    def generate_object_list_for_questions(self, image, questions):
        """Generate object list relevant to all questions for this image"""
        try:
            # Combine all questions for this image
            combined_questions = " | ".join(questions)
            
            prompt = f"""USER: <image> For the provided image and its associated questions, generate a scene graph in JSON format that includes the following:

1. Objects that are relevant to answering any of the questions
2. Object attributes that are relevant to answering the questions
3. Object relationships that are relevant to answering the questions

Questions: "{combined_questions}"

Focus on accuracy and relevance to the questions. Include objects that help answer any of these questions.
Object List(in JSON format only):

A:"""
            
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,  # Increased for multiple questions
                    do_sample=True,
                    temperature=0.15,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            if "ASSISTANT:" in generated_text:
                object_list_text = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                object_list_text = generated_text.replace(prompt.replace("<image>", ""), "").strip()
            
            object_list_text = self._clean_scene_graph_output(object_list_text)
            
            return object_list_text if object_list_text else '{"objects": []}'
            
        except Exception as e:
            return f'{{"error": "Error generating object list: {str(e)}"}}'

def load_whoops_dataset():
    """Load WHOOPS dataset from HuggingFace"""
    print("Loading WHOOPS dataset from HuggingFace...")
    
    try:
        dataset = load_dataset("nlphuji/whoops")
        print("Successfully loaded WHOOPS dataset")
        
        # Use both train and test splits
        all_data = []
        for split in ['train', 'test']:
            if split in dataset:
                split_data = dataset[split]
                print(f"{split} split has {len(split_data)} samples")
                all_data.extend(list(split_data))
        
        print(f"Total samples across all splits: {len(all_data)}")
        return all_data
        
    except Exception as e:
        print(f"Failed to load WHOOPS dataset: {e}")
        raise

def parse_qa_pairs(qa_pairs_str):
    """Parse question-answering pairs from string format"""
    if not qa_pairs_str:
        return []
    
    try:
        # Parse the string representation of the list
        qa_pairs = ast.literal_eval(qa_pairs_str) if isinstance(qa_pairs_str, str) else qa_pairs_str
        return qa_pairs if isinstance(qa_pairs, list) else []
    except:
        return []

def find_valid_whoops_samples(samples, max_needed=None):
    """Find valid image samples from WHOOPS dataset"""
    valid_data = []
    
    max_samples = max_needed if max_needed else len(samples)
    print(f"Finding valid images from {len(samples)} WHOOPS samples (max: {max_samples})...")
    
    for i, sample in enumerate(samples):
        try:
            if len(valid_data) >= max_samples:
                print(f"Reached maximum samples ({max_samples}), stopping search...")
                break
                
            # Check if image exists and is valid
            image = sample.get('image')
            image_id = sample.get('image_id')
            
            if image is None or image_id is None:
                continue
                
            if isinstance(image, Image.Image):
                try:
                    _ = image.size  # Test if image is accessible
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Parse QA pairs
                    qa_pairs_str = sample.get('question_answering_pairs', '[]')
                    qa_pairs = parse_qa_pairs(qa_pairs_str)
                    
                    if not qa_pairs:  # Skip samples without questions
                        continue
                    
                    # Create sample with processed data
                    processed_sample = {
                        'image': image,
                        'image_id': image_id,
                        'qa_pairs': qa_pairs,
                        'designer_explanation': sample.get('designer_explanation', ''),
                        'selected_caption': sample.get('selected_caption', ''),
                        'commonsense_category': sample.get('commonsense_category', 'unknown'),
                        'original_index': i
                    }
                    
                    valid_data.append(processed_sample)
                        
                except Exception as img_error:
                    if i < 10:  # Only log first few errors
                        print(f"Image processing error for sample {i}: {img_error}")
                    continue
                
        except Exception as e:
            if i < 10:  # Only log first few errors
                print(f"Error checking sample {i}: {e}")
            continue
    
    print(f"Found {len(valid_data)} valid WHOOPS samples")
    return valid_data

def process_whoops_data(num_samples=None):
    """Process WHOOPS data and generate scene graphs"""
    
    # Load WHOOPS dataset
    all_samples = load_whoops_dataset()
    
    if not all_samples:
        print("No samples found in WHOOPS dataset!")
        return []
    
    # Find valid samples
    valid_samples = find_valid_whoops_samples(all_samples, max_needed=num_samples)
    
    if len(valid_samples) == 0:
        print("No valid image samples found in WHOOPS dataset!")
        return []
    
    print(f"Processing {len(valid_samples)} valid WHOOPS samples...")
    
    # Initialize scene graph generator
    generator = WHOOPSSceneGraphGenerator()
    
    results = []
    
    print(f"Generating scene graphs for {len(valid_samples)} samples...")
    
    for i, sample in enumerate(tqdm(valid_samples, desc="Generating scene graphs")):
        try:
            image = sample['image']
            image_id = sample['image_id']
            qa_pairs = sample['qa_pairs']
            designer_explanation = sample['designer_explanation']
            selected_caption = sample['selected_caption']
            commonsense_category = sample['commonsense_category']
            original_idx = sample['original_index']
            
            if image is None:
                print(f"No image found for sample {image_id}")
                continue
            
            # Ensure image is RGB
            if hasattr(image, 'mode') and image.mode != 'RGB':
                image = image.convert('RGB')
            
            if i % 10 == 0:  # Reduce logging frequency
                print(f"Processing sample {i+1}/{len(valid_samples)} (Image ID: {image_id})")
            
            # Extract all questions for this image
            questions = []
            qa_data = []
            for qa_pair in qa_pairs:
                if isinstance(qa_pair, list) and len(qa_pair) >= 2:
                    question = qa_pair[0]
                    answer = qa_pair[1]
                    questions.append(question)
                    qa_data.append({"question": question, "answer": answer})
            
            if not questions:
                print(f"No valid questions found for sample {image_id}")
                continue
            
            # Generate global scene graph
            global_scene_graph = generator.generate_global_scene_graph(image)
            
            # Generate object list considering all questions for this image
            object_list = generator.generate_object_list_for_questions(image, questions)
            
            # Store result
            result = {
                'image_id': image_id,
                'original_dataset_index': original_idx,
                'qa_pairs': qa_data,
                'questions': questions,  # All questions for easy access
                'designer_explanation': designer_explanation,
                'selected_caption': selected_caption,
                'commonsense_category': commonsense_category,
                'global_scene_graph': global_scene_graph,
                'object_list': object_list,
                'num_questions': len(questions)
            }
            
            results.append(result)
            
            # Save intermediate results every 50 samples
            if (i + 1) % 50 == 0:
                print(f"Saving intermediate results ({i+1} samples processed)...")
                with open('results_whoops_intermediate.json', 'w') as f:
                    json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"Error processing sample {image_id if 'image_id' in locals() else i}: {e}")
            continue
    
    # Save final results
    print("Saving final results to results_whoops.json...")
    with open('results_whoops.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create mapping and summary data
    mapping_data = {
        'total_samples_processed': len(results),
        'image_ids': [r['image_id'] for r in results],
        'total_questions': sum(r['num_questions'] for r in results),
        'dataset': 'WHOOPS',
        'splits_used': ['train', 'test'],
        'avg_questions_per_image': sum(r['num_questions'] for r in results) / len(results) if results else 0
    }
    
    with open('whoops_index_mapping.json', 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"Successfully processed {len(results)} samples!")
    print(f"Total questions across all samples: {mapping_data['total_questions']}")
    print(f"Average questions per image: {mapping_data['avg_questions_per_image']:.2f}")
    print(f"Results saved to: results_whoops.json")
    print(f"Index mapping saved to: whoops_index_mapping.json")
    
    return results

def main():
    """Main function"""
    print("WHOOPS Scene Graph Generation")
    print("=" * 60)
    
    try:
        # Process all WHOOPS samples (remove num_samples parameter to process all)
        results = process_whoops_data(num_samples=None)  # Set to specific number to limit
        
        print(f"\nProcessing complete!")
        print(f"Total samples processed: {len(results)}")
        print(f"Results saved to: results_whoops.json")
        print(f"Index mapping saved to: whoops_index_mapping.json")
        
        # Show sample structure for verification
        if results:
            print(f"\nSample result structure:")
            sample_result = results[0]
            for key, value in sample_result.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                elif isinstance(value, list) and key == 'qa_pairs':
                    print(f"  {key}: {len(value)} Q&A pairs")
                    if value:
                        print(f"    First Q&A: {value[0]}")
                else:
                    print(f"  {key}: {value}")
        
        # Show category distribution
        if results:
            categories = {}
            for result in results:
                cat = result.get('commonsense_category', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            print(f"\nCommonsense category distribution:")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cat}: {count}")
                
        # Show questions per image distribution
        if results:
            question_counts = [r['num_questions'] for r in results]
            print(f"\nQuestions per image distribution:")
            print(f"  Min: {min(question_counts)}")
            print(f"  Max: {max(question_counts)}")
            print(f"  Average: {sum(question_counts)/len(question_counts):.2f}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()