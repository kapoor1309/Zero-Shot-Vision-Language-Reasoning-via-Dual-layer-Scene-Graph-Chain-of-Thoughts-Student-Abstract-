import torch
import json
import re
import os
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
from datetime import datetime
from PIL import Image

class MMBenchQuerySpecificSGGenerator:
    def __init__(self):
        self.processor = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load LLaVA model"""
        try:
            print("Loading LLaVA model...")
            model_name = "llava-hf/llava-1.5-13b-hf"
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _safe_json_loads(self, json_str):
        """Safely load JSON with error handling"""
        if not json_str or not isinstance(json_str, str):
            return None
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                fixed = re.sub(r',\s*}', '}', json_str)
                fixed = re.sub(r',\s*]', ']', fixed)
                fixed = fixed.replace('\\_', '_')
                return json.loads(fixed)
            except:
                return None
    
    def _extract_json_from_response(self, text):
        """Enhanced JSON extraction from model response"""
        if not text:
            return None
        
        text = text.strip()
        
        strategies = [
            r'\{\s*"objects"\s*:\s*\[[^\]]*\]\s*,\s*"relationships"\s*:\s*\[[^\]]*\]\s*\}',
            r'\{\s*"objects"\s*:\s*\[.*?\]\s*(?:,\s*"relationships"\s*:\s*\[.*?\])?\s*\}',
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
            r'\{.*?\}',
        ]
        
        for strategy in strategies:
            matches = re.findall(strategy, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                cleaned_match = self._clean_json_string(match)
                
                parsed = self._safe_json_loads(cleaned_match)
                if parsed and isinstance(parsed, dict):
                    if 'objects' not in parsed:
                        parsed['objects'] = []
                    if 'relationships' not in parsed:
                        parsed['relationships'] = []
                    return parsed
        
        return None
    
    def _clean_json_string(self, json_str):
        """Clean JSON string to make it parseable"""
        if not json_str:
            return json_str
        
        json_str = re.sub(r'\s+', ' ', json_str)
        
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        first_brace = json_str.find('{')
        if first_brace > 0:
            json_str = json_str[first_brace:]
        
        last_brace = json_str.rfind('}')
        if last_brace >= 0:
            json_str = json_str[:last_brace + 1]
        
        return json_str.strip()
    
    def _extract_and_clean_object_list(self, raw_object_list):
        """Extract and clean object list from MMBench format"""
        if not raw_object_list:
            return '{"objects": []}'
        
        if isinstance(raw_object_list, dict):
            if 'objects' in raw_object_list:
                return json.dumps(raw_object_list)
            else:
                return '{"objects": []}'
        
        if isinstance(raw_object_list, str):
            cleaned = raw_object_list.replace('\\n', '\n').replace('\\"', '"').replace('\\_', '_')
            
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    if 'objects' not in parsed:
                        parsed['objects'] = []
                    return json.dumps(parsed)
            except:
                pass
            
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                try:
                    json_content = json_match.group(0)
                    parsed = json.loads(json_content)
                    if isinstance(parsed, dict):
                        if 'objects' not in parsed:
                            parsed['objects'] = []
                        return json.dumps(parsed)
                except:
                    pass
        
        return '{"objects": []}'
    
    def _validate_result_object(self, obj):
        """Validate that object has required MMBench fields"""
        return (isinstance(obj, dict) and 
                'original_dataset_index' in obj and 
                'question' in obj and
                obj.get('original_dataset_index') is not None and 
                obj.get('question'))
    
    def generate_query_specific_sg(self, image, question, object_list, global_scene_graph=""):
        """Generate query-specific scene graph for MMBench"""
        try:
            readable_objects = "- No specific objects detected, analyze the image directly"
            extracted_objects = []
            
            try:
                objects_data = json.loads(object_list)
                if 'objects' in objects_data and objects_data['objects']:
                    objects = objects_data['objects']
                    object_names = []
                    
                    for obj in objects[:15]:
                        if isinstance(obj, dict):
                            name = obj.get('name', 'Unknown')
                            obj_type = obj.get('type', '')
                            material = obj.get('material', '')
                            position = obj.get('position', '')
                            
                            if name == "Object Name" or "Object Name" in name:
                                continue
                            
                            extracted_objects.append(obj)
                            
                            obj_desc = f"- {name}"
                            if obj_type and obj_type != "Object Type":
                                obj_desc += f" ({obj_type})"
                            if material:
                                obj_desc += f" [material: {material}]"
                            if position and isinstance(position, list):
                                obj_desc += f" [position: {position}]"
                            
                            object_names.append(obj_desc)
                        elif isinstance(obj, str):
                            if obj != "Object Name":
                                object_names.append(f"- {obj}")
                                extracted_objects.append({"name": obj})
                    
                    if object_names:
                        readable_objects = '\n'.join(object_names)
                    else:
                        readable_objects = "- No specific objects detected, analyze the image directly"
                        
            except Exception as e:
                print(f"Error parsing object list: {e}")
                readable_objects = "- Error parsing object list, analyze the image directly"
            
            if extracted_objects:
                prompt = f"""USER: <image>
Analyze the image and generate a scene graph in valid JSON format for the question below.

Available Objects:
{readable_objects}

Question: "{question}"

Generate a JSON response with this exact structure:
{{
    "objects": [
        {{"name": "object_name", "attributes": ["attr1", "attr2"]}}
    ],
    "relationships": [
        {{"subject": "obj1", "predicate": "relation", "object": "obj2"}}
    ]
}}

Focus only on objects and relationships that help answer the question. Ensure the JSON is valid and complete.

A:
{{"""
            else:
                prompt = f"""USER: <image>
Analyze the image and generate a scene graph in valid JSON format for the question below.

Question: "{question}"

Generate a JSON response with this exact structure:
{{
    "objects": [
        {{"name": "object_name", "attributes": ["attr1", "attr2"]}}
    ],
    "relationships": [
        {{"subject": "obj1", "predicate": "relation", "object": "obj2"}}
    ]
}}

Focus only on objects and relationships that help answer the question. Ensure the JSON is valid and complete.

A: Looking at this image and the question "{question}", I'll generate a scene graph focusing on relevant elements.

{{"""
            
            inputs = self.processor(text=prompt, images=image, return_tensors="pt", padding=True)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            if "ASSISTANT:" in generated_text:
                response_parts = generated_text.split("ASSISTANT:")
                if len(response_parts) > 1:
                    sg_text = response_parts[-1].strip()
                else:
                    sg_text = generated_text
            else:
                prompt_clean = prompt.replace("USER: <image>", "").strip()
                sg_text = generated_text.replace(prompt_clean, "").strip()
            
            print(f"DEBUG - Raw model output: {sg_text[:200]}...")
            
            parsed_sg = self._extract_json_from_response(sg_text)
            
            if parsed_sg:
                print(f"DEBUG - Successfully extracted JSON with {len(parsed_sg.get('objects', []))} objects")
                return json.dumps(parsed_sg, indent=2)
            else:
                print(f"DEBUG - Failed to extract valid JSON, using fallback")
                fallback_sg = {
                    "objects": [],
                    "relationships": []
                }
                
                try:
                    potential_objects = re.findall(r'"([^"]+)"', sg_text)
                    for obj_name in potential_objects[:5]:
                        if len(obj_name) > 2 and obj_name.lower() not in ['objects', 'relationships', 'name', 'attributes']:
                            fallback_sg['objects'].append({
                                "name": obj_name,
                                "attributes": []
                            })
                except:
                    pass
                
                return json.dumps(fallback_sg, indent=2)
                
        except Exception as e:
            print(f"Error generating scene graph: {e}")
            error_sg = {
                "objects": [],
                "relationships": [],
                "error": f"Generation failed: {str(e)}"
            }
            return json.dumps(error_sg, indent=2)
    
    def load_and_clean_mmbench_results(self, filepath="results_mmbench.json"):
        """Load and clean MMBench results"""
        print(f"Loading MMBench results from {filepath}...")
        
        if not os.path.exists(filepath):
            print(f"{filepath} file not found!")
            return []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            if not isinstance(results, list):
                print("Expected a list of results")
                return []
            
            cleaned_results = []
            for result in results:
                if self._validate_result_object(result):
                    result['object_list'] = self._extract_and_clean_object_list(
                        result.get('object_list', '')
                    )
                    cleaned_results.append(result)
            
            print(f"Loaded and validated {len(cleaned_results)} results")
            return cleaned_results
            
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            return []
    
    def load_mmbench_images(self):
        """Load MMBench dataset images with optimized lookup"""
        print("Loading MMBench dataset...")
        
        try:
            dataset = load_dataset("lmms-lab/MMBench_EN")
            dev_data = dataset['dev']
            print(f"Loaded MMBench dev set with {len(dev_data)} samples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return {}
        
        image_lookup = {}
        print("Building image lookup...")
        
        for i, sample in enumerate(tqdm(dev_data, desc="Indexing images")):
            try:
                dataset_idx = str(i)
                
                image = sample.get('image')
                if image and hasattr(image, 'convert'):
                    image_lookup[dataset_idx] = image.convert('RGB')
                elif image is not None:
                    image_lookup[dataset_idx] = image
                        
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        print(f"Indexed {len(image_lookup)} images")
        return image_lookup
    
    def process_query_specific_sg(self, results_file="results_mmbench.json", max_samples=None, debug=False):
        """Generate query-specific scene graphs from MMBench results"""
        
        results = self.load_and_clean_mmbench_results(results_file)
        
        if not results:
            print("No valid results found! Please check your results file.")
            return []
        
        print(f"Found {len(results)} MMBench samples to process")
        
        if max_samples:
            results = results[:max_samples]
            print(f"Limited to {len(results)} samples")
        
        image_lookup = self.load_mmbench_images()
        
        if not image_lookup:
            print("No images loaded from MMBench dataset!")
            return []
        
        query_sg_results = []
        missing_images = 0
        processing_errors = 0
        successful_generations = 0
        
        print("Generating query-specific scene graphs for MMBench...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"mmbench_query_sg_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        for i, result in enumerate(tqdm(results, desc="Processing MMBench samples")):
            try:
                original_dataset_index = str(result['original_dataset_index'])
                question = result['question']
                category = result.get('category', 'unknown')
                l2_category = result.get('l2_category', 'unknown')
                answer = result.get('answer', 'unknown')
                options = result.get('options', {})
                object_list = result.get('object_list', '{"objects": []}')
                global_scene_graph = result.get('global_scene_graph', '')
                
                if original_dataset_index not in image_lookup:
                    print(f"Image at index {original_dataset_index} not found in dataset")
                    missing_images += 1
                    continue
                
                image = image_lookup[original_dataset_index]
                
                query_sg = self.generate_query_specific_sg(
                    image, question, object_list, global_scene_graph
                )
                
                try:
                    parsed_sg = json.loads(query_sg)
                    if 'objects' in parsed_sg or 'relationships' in parsed_sg:
                        successful_generations += 1
                        if debug and i < 5:
                            print(f"Sample {i} - Generated SG: {query_sg[:200]}...")
                except:
                    print(f"Warning: Invalid JSON generated for sample {i}")
                
                sg_result = {
                    'original_dataset_index': original_dataset_index,
                    'question': question,
                    'category': category,
                    'l2_category': l2_category,
                    'answer': answer,
                    'options': options,
                    'object_list': object_list,
                    'global_scene_graph': global_scene_graph,
                    'query_specific_scene_graph': query_sg,
                    'timestamp': datetime.now().isoformat()
                }
                
                query_sg_results.append(sg_result)
                
                if (i + 1) % 50 == 0:
                    intermediate_file = os.path.join(results_dir, f'mmbench_sg_intermediate_{i+1}.json')
                    with open(intermediate_file, 'w') as f:
                        json.dump(query_sg_results, f, indent=2)
                    print(f"Saved intermediate results ({i+1} processed, {successful_generations} successful)")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                processing_errors += 1
                continue
        
        final_results_file = os.path.join(results_dir, 'mmbench_sg_final.json')
        with open(final_results_file, 'w') as f:
            json.dump(query_sg_results, f, indent=2)
        
        with open('mmbench_sg.json', 'w') as f:
            json.dump(query_sg_results, f, indent=2)
        
        summary = {
            'timestamp': timestamp,
            'dataset': 'MMBench_EN',
            'split': 'dev',
            'total_input_samples': len(results),
            'successful_generations': successful_generations,
            'total_processed': len(query_sg_results),
            'missing_images': missing_images,
            'processing_errors': processing_errors,
            'success_rate': successful_generations / len(results) if results else 0,
            'processing_rate': len(query_sg_results) / len(results) if results else 0,
            'results_directory': results_dir,
            'categories_processed': list(set([r.get('category', 'unknown') for r in results]))
        }
        
        summary_file = os.path.join(results_dir, 'mmbench_processing_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Successfully processed {len(query_sg_results)} samples!")
        print(f"Valid scene graphs generated: {successful_generations}")
        print(f"Missing images: {missing_images}")
        print(f"Processing errors: {processing_errors}")
        print(f"Results saved to: {final_results_file}")
        print(f"Also saved to: mmbench_sg.json")
        
        if query_sg_results:
            categories = {}
            for result in query_sg_results:
                cat = result.get('category', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
            
            print(f"\nCategory distribution in processed samples:")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cat}: {count}")
        
        return query_sg_results

def main():
    """Main function for MMBench query-specific scene graph generation"""
    print("MMBench Query-Specific Scene Graph Generation")
    print("=" * 60)
    
    results_file = "results_mmbench.json"
    max_samples = None
    debug_mode = True
    
    print(f"Configuration:")
    print(f"  Results file: {results_file}")
    print(f"  Max samples: {max_samples or 'All'}")
    print(f"  Debug mode: {debug_mode}")
    print(f"  Dataset: MMBench_EN")
    
    try:
        generator = MMBenchQuerySpecificSGGenerator()
        
        results = generator.process_query_specific_sg(
            results_file=results_file,
            max_samples=max_samples,
            debug=debug_mode
        )
        
        print(f"\nMMBench processing complete!")
        print(f"Total samples processed: {len(results)}")
        
        if results:
            print(f"\nSample result structure:")
            sample = results[0]
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
            
            empty_count = 0
            valid_count = 0
            error_count = 0
            
            for result in results:
                try:
                    sg = json.loads(result['query_specific_scene_graph'])
                    if 'error' in sg:
                        error_count += 1
                    elif 'objects' in sg and sg['objects']:
                        valid_count += 1
                    else:
                        empty_count += 1
                        
                except:
                    error_count += 1
            
            print(f"\nScene Graph Quality:")
            print(f"  Valid scene graphs (with objects): {valid_count}")
            print(f"  Empty scene graphs: {empty_count}")
            print(f"  Error scene graphs: {error_count}")
            print(f"  Success rate: {valid_count/len(results)*100:.1f}%")
            
            if empty_count > len(results) * 0.8:
                print("\nWARNING: High number of empty scene graphs detected!")
                print("Consider adjusting the generation parameters or prompt.")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()