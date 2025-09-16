import torch
import json
import random
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import re

class SEEDSceneGraphGenerator:
    def __init__(self):
        self.processor = None
        self.model = None
        self.load_model()
    
    def load_model(self):
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
        text = text.strip()
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return text
    
    def generate_global_scene_graph(self, image):
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
        try:
            prompt = f"""USER: <image> For the provided image and its associated question, generate a scene graph in JSON format that includes the following:

1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question
3. Object relationships that are relevant to answering the questuion

Question: "{question}"


Focus on accuracy and relevance to the question. 
Scene Graph (in JSON format only):

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

def load_seed_dataset(subset_size=2000):
    print("Loading SEED-Bench dataset from HuggingFace...")
    
    try:
        dataset = load_dataset("lmms-lab/SEED-Bench")
        print("Successfully loaded SEED-Bench dataset")
        
        test_data = dataset['test']
        print(f"Full test split has {len(test_data)} samples")
        
        print(f"Taking random subset of {subset_size} samples...")
        total_samples = len(test_data)
        if total_samples > subset_size:
            random_indices = random.sample(range(total_samples), subset_size)
            subset_data = test_data.select(random_indices)
        else:
            subset_data = test_data
            
        print(f"Working with subset of {len(subset_data)} samples")
        
        print("Filtering subset for image data type...")
        image_data = subset_data.filter(lambda x: x['data_type'] == 'image')
        print(f"Found {len(image_data)} image samples in subset")
        
        return image_data
        
    except Exception as e:
        print(f"Failed to load SEED-Bench dataset: {e}")
        raise

def find_valid_samples(image_data, max_needed=1000):
    valid_indices = []
    print(f"Finding {max_needed} valid images from {len(image_data)} image samples...")
    
    for i in range(len(image_data)):
        try:
            sample = image_data[i]
            
            if 'image' in sample and sample['image'] is not None:
                image_list = sample['image']
                
                if isinstance(image_list, list) and len(image_list) > 0:
                    image = image_list[0]
                    
                    if isinstance(image, Image.Image):
                        try:
                            _ = image.size
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            valid_indices.append(i)
                        except Exception as img_error:
                            if i < 10:
                                print(f"Image processing error for sample {i}: {img_error}")
                            continue
            
            if len(valid_indices) >= max_needed:
                print(f"Found {max_needed} valid samples, stopping search...")
                break
                
        except Exception as e:
            if i < 10:
                print(f"Error checking sample {i}: {e}")
            continue
    
    print(f"Found {len(valid_indices)} valid samples out of {len(image_data)} image samples checked")
    return valid_indices

def process_seed_data(num_samples=1000, subset_size=2000):
    image_data = load_seed_dataset(subset_size=subset_size)
    
    if len(image_data) == 0:
        print("No image data found in subset! Try increasing subset_size.")
        return []
    
    valid_indices = find_valid_samples(image_data, max_needed=num_samples)
    
    if len(valid_indices) == 0:
        print("No valid samples found! Check dataset format.")
        return []
    
    actual_samples = min(len(valid_indices), num_samples)
    sample_indices = valid_indices[:actual_samples]
    
    print(f"Processing {len(sample_indices)} valid samples...")
    
    generator = SEEDSceneGraphGenerator()
    
    results = []
    
    print(f"Generating scene graphs for {len(sample_indices)} samples...")
    
    for i, idx in enumerate(tqdm(sample_indices, desc="Generating scene graphs")):
        try:
            sample = image_data[idx]
            
            image_list = sample.get('image', [])
            if not image_list or len(image_list) == 0:
                print(f"No image found for sample {idx}")
                continue
                
            image = image_list[0]
            question = sample.get('question', 'What do you see in this image?')
            question_id = sample.get('question_id', f'seed_{idx}')
            data_id = sample.get('data_id', f'data_{idx}')
            
            if hasattr(image, 'mode') and image.mode != 'RGB':
                image = image.convert('RGB')
            
            if i % 10 == 0:
                print(f"Processing sample {i+1}/{len(sample_indices)} (ID: {question_id})")
            
            global_scene_graph = generator.generate_global_scene_graph(image)
            
            object_list = generator.generate_object_list(image, question)
            
            result = {
                'unique_image_id': data_id,
                'question_id': question_id,
                'question': question,
                'data_type': sample.get('data_type', 'image'),
                'global_scene_graph': global_scene_graph,
                'object_list': object_list
            }
            
            results.append(result)
            
            if (i + 1) % 50 == 0:
                print(f"Saving intermediate results ({i+1} samples processed)...")
                with open('results_seed_intermediate.json', 'w') as f:
                    json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    print("Saving final results to results_seed.json...")
    with open('results_seed.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Successfully processed {len(results)} samples!")
    print(f"Results saved to: results_seed.json")
    
    return results

def main():
    print("SEED-Bench Scene Graph Generation")
    print("=" * 50)
    
    try:
        results = process_seed_data(num_samples=1000, subset_size=2000)
        
        print(f"\nProcessing complete!")
        print(f"Total samples processed: {len(results)}")
        print(f"Results saved to: results_seed.json")
        
        if results:
            print(f"\nSample result structure:")
            sample_result = results[0]
            for key, value in sample_result.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()