import torch
import json
import random
import pandas as pd
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import re

class MMBenchSceneGraphGenerator:
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
    
    def generate_object_list(self, image, question):
        """Generate object list relevant to the question"""
        try:
            prompt = f"""USER: <image> For the provided image and its associated question, generate a scene graph in JSON format that includes the following:

1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the question

Question: "{question}"

Focus on accuracy and relevance to the question. 
Object List(in JSON format only):

A:"""
            
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
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

def load_mmbench_dataset():
    """Load MMBench dataset and filter for samples with NaN hint"""
    print("Loading MMBench dataset from HuggingFace...")
    
    try:
        dataset = load_dataset("lmms-lab/MMBench_EN")
        print("Successfully loaded MMBench dataset")
        
        dev_data = dataset['dev']
        print(f"Full dev split has {len(dev_data)} samples")
        
        return dev_data
        
    except Exception as e:
        print(f"Failed to load MMBench dataset: {e}")
        raise

def filter_nan_hints(dataset_split):
    """Filter dataset for samples where hint column is NaN and return their indices"""
    print("Filtering for samples with NaN hints...")
    
    valid_samples = []
    original_indices = []
    
    for i in range(len(dataset_split)):
        sample = dataset_split[i]
        hint = sample.get('hint', None)
        
        if hint is None or hint == '' or (isinstance(hint, str) and hint.lower() == 'nan'):
            valid_samples.append(sample)
            original_indices.append(i)
    
    print(f"Found {len(valid_samples)} samples with NaN hints out of {len(dataset_split)} total samples")
    return valid_samples, original_indices

def find_valid_image_samples(samples, original_indices, max_needed=1000):
    """Find valid image samples from the filtered list"""
    valid_data = []
    valid_indices = []
    
    print(f"Finding {max_needed} valid images from {len(samples)} samples with NaN hints...")
    
    for i, (sample, orig_idx) in enumerate(zip(samples, original_indices)):
        try:
            if 'image' in sample and sample['image'] is not None:
                image = sample['image']
                
                if isinstance(image, Image.Image):
                    try:
                        _ = image.size
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        sample_with_index = dict(sample)
                        sample_with_index['original_index'] = orig_idx
                        
                        valid_data.append(sample_with_index)
                        valid_indices.append(orig_idx)
                        
                    except Exception as img_error:
                        if i < 10:
                            print(f"Image processing error for sample {orig_idx}: {img_error}")
                        continue
            
            if len(valid_data) >= max_needed:
                print(f"Found {max_needed} valid samples, stopping search...")
                break
                
        except Exception as e:
            if i < 10:
                print(f"Error checking sample {orig_idx}: {e}")
            continue
    
    print(f"Found {len(valid_data)} valid image samples with NaN hints")
    return valid_data, valid_indices

def process_mmbench_data(num_samples=1000):
    """Process MMBench data and generate scene graphs for samples with NaN hints"""
    
    dev_data = load_mmbench_dataset()
    
    nan_hint_samples, original_indices = filter_nan_hints(dev_data)
    
    if len(nan_hint_samples) == 0:
        print("No samples found with NaN hints!")
        return []
    
    valid_samples, valid_indices = find_valid_image_samples(
        nan_hint_samples, original_indices, max_needed=num_samples
    )
    
    if len(valid_samples) == 0:
        print("No valid image samples found with NaN hints!")
        return []
    
    actual_samples = min(len(valid_samples), num_samples)
    samples_to_process = valid_samples[:actual_samples]
    
    print(f"Processing {len(samples_to_process)} valid samples with NaN hints...")
    
    generator = MMBenchSceneGraphGenerator()
    
    results = []
    
    print(f"Generating scene graphs for {len(samples_to_process)} samples...")
    
    for i, sample in enumerate(tqdm(samples_to_process, desc="Generating scene graphs")):
        try:
            image = sample.get('image')
            question = sample.get('question', 'What do you see in this image?')
            original_idx = sample.get('original_index', i)
            
            category = sample.get('category', 'unknown')
            l2_category = sample.get('l2-category', 'unknown')
            answer = sample.get('answer', 'unknown')
            
            option_a = sample.get('A', '')
            option_b = sample.get('B', '')
            option_c = sample.get('C', '')
            option_d = sample.get('D', '')
            
            if image is None:
                print(f"No image found for sample {original_idx}")
                continue
            
            if hasattr(image, 'mode') and image.mode != 'RGB':
                image = image.convert('RGB')
            
            if i % 10 == 0:
                print(f"Processing sample {i+1}/{len(samples_to_process)} (Original Index: {original_idx})")
            
            global_scene_graph = generator.generate_global_scene_graph(image)
            
            object_list = generator.generate_object_list(image, question)
            
            result = {
                'original_dataset_index': original_idx,
                'question': question,
                'category': category,
                'l2_category': l2_category,
                'answer': answer,
                'hint': sample.get('hint', 'nan'),
                'options': {
                    'A': option_a,
                    'B': option_b,
                    'C': option_c,
                    'D': option_d
                },
                'global_scene_graph': global_scene_graph,
                'object_list': object_list
            }
            
            results.append(result)
            
            if (i + 1) % 50 == 0:
                print(f"Saving intermediate results ({i+1} samples processed)...")
                with open('results_mmbench_intermediate.json', 'w') as f:
                    json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"Error processing sample {original_idx}: {e}")
            continue
    
    print("Saving final results to results_mmbench.json...")
    with open('results_mmbench.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    mapping_data = {
        'total_samples_processed': len(results),
        'original_indices': [r['original_dataset_index'] for r in results],
        'filter_criteria': 'hint == nan',
        'dataset': 'MMBench_EN',
        'split': 'dev'
    }
    
    with open('mmbench_index_mapping.json', 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"Successfully processed {len(results)} samples!")
    print(f"Results saved to: results_mmbench.json")
    print(f"Index mapping saved to: mmbench_index_mapping.json")
    
    return results

def main():
    """Main function"""
    print("MMBench Scene Graph Generation (NaN Hints Only)")
    print("=" * 60)
    
    try:
        results = process_mmbench_data(num_samples=1000)
        
        print(f"\nProcessing complete!")
        print(f"Total samples processed: {len(results)}")
        print(f"Results saved to: results_mmbench.json")
        print(f"Index mapping saved to: mmbench_index_mapping.json")
        
        if results:
            print(f"\nSample result structure:")
            sample_result = results[0]
            for key, value in sample_result.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
                    
        if results:
            categories = {}
            for result in results:
                cat = result.get('category', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            print(f"\nCategory distribution:")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cat}: {count}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()