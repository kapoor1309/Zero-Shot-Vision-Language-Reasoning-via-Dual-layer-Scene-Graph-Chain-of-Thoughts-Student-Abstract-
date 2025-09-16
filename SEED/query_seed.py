import torch
import json
import re
import os
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
from datetime import datetime

class ImprovedQuerySpecificSGGenerator:
    def __init__(self):
        self.processor = None
        self.model = None
        self.load_model()
    
    def load_model(self):
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
    
    def _extract_json_from_text(self, text):
        if not text:
            return None
        
        json_patterns = [
            r'\{[^{}]*"objects"[^{}]*\[[^\]]*\][^{}]*\}',
            r'\{.*?"objects".*?\}',
            r'\{[^{}]*\}',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                parsed = self._safe_json_loads(match)
                if parsed and 'objects' in parsed:
                    return parsed
        
        return None
    
    def _parse_malformed_json_file(self, filepath):
        objects = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                content_cleaned = re.sub(r',\s*\]', ']', content)
                content_cleaned = re.sub(r',\s*\}', '}', content_cleaned)
                
                if content_cleaned.strip().startswith('['):
                    objects = json.loads(content_cleaned)
                    print(f"Loaded JSON as array: {len(objects)} items")
                    return objects
                    
            except json.JSONDecodeError:
                print("Trying line-by-line parsing...")
            
            lines = content.split('\n')
            current_obj = ""
            brace_count = 0
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if line.startswith('{') and line.endswith('}'):
                    try:
                        line_cleaned = re.sub(r',\s*}', '}', line)
                        line_cleaned = re.sub(r',\s*]', ']', line_cleaned)
                        line_cleaned = line_cleaned.replace('\\_', '_')
                        
                        obj = json.loads(line_cleaned)
                        if self._validate_result_object(obj):
                            objects.append(obj)
                        continue
                    except:
                        pass
                
                if '{' in line or current_obj:
                    current_obj += line + " "
                    brace_count += line.count('{') - line.count('}')
                    
                    if brace_count == 0 and current_obj.strip():
                        obj_str = current_obj.strip().rstrip(',')
                        parsed = self._safe_json_loads(obj_str)
                        
                        if parsed and self._validate_result_object(parsed):
                            objects.append(parsed)
                        else:
                            extracted = self._extract_json_from_text(obj_str)
                            if extracted and self._validate_result_object(extracted):
                                objects.append(extracted)
                        
                        current_obj = ""
                        brace_count = 0
            
            if not objects:
                print("Trying pattern matching...")
                pattern = r'\{[^{}]*"unique_image_id"[^{}]*"question"[^{}]*\}'
                matches = re.findall(pattern, content, re.DOTALL)
                
                for match in matches:
                    try:
                        match_cleaned = re.sub(r',\s*}', '}', match)
                        match_cleaned = re.sub(r',\s*]', ']', match_cleaned)
                        match_cleaned = match_cleaned.replace('\\_', '_')
                        
                        obj = json.loads(match_cleaned)
                        if self._validate_result_object(obj):
                            objects.append(obj)
                    except:
                        continue
            
            print(f"Parsed {len(objects)} objects from malformed JSON")
            return objects
            
        except Exception as e:
            print(f"Error parsing file {filepath}: {e}")
            return []
    
    def _validate_result_object(self, obj):
        return (isinstance(obj, dict) and 
                'unique_image_id' in obj and 
                'question' in obj and
                obj.get('unique_image_id') and 
                obj.get('question'))
    
    def _extract_and_clean_object_list(self, raw_object_list):
        if not raw_object_list:
            return '{"objects": []}'
        
        if isinstance(raw_object_list, dict):
            if 'objects' in raw_object_list:
                return json.dumps(raw_object_list)
            else:
                return '{"objects": []}'
        
        if isinstance(raw_object_list, str):
            cleaned = raw_object_list.replace('\\n', '\n').replace('\\"', '"')
            
            if "A:" in cleaned:
                after_a = cleaned.split("A:")[-1].strip()
                
                for letter in ['B:', 'C:', 'D:']:
                    if letter in after_a:
                        after_a = after_a.split(letter)[0].strip()
                
                try:
                    start = after_a.find('{')
                    if start != -1:
                        brace_count = 0
                        end = len(after_a)
                        
                        for i, char in enumerate(after_a[start:], start):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end = i + 1
                                    break
                        
                        json_part = after_a[start:end]
                        obj = json.loads(json_part)
                        
                        if isinstance(obj, dict) and 'name' in obj:
                            return json.dumps({"objects": [obj]})
                        elif isinstance(obj, dict) and 'objects' in obj:
                            return json.dumps(obj)
                
                except Exception as e:
                    print(f"Error parsing JSON after A:: {e}")
                    print(f"Attempted to parse: {after_a[:200]}...")
                
                json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]*"[^{}]*\}'
                matches = re.findall(json_pattern, after_a, re.DOTALL)
                
                if matches:
                    for match in matches:
                        try:
                            obj = json.loads(match)
                            if isinstance(obj, dict) and 'name' in obj and obj['name'] != "Object Name":
                                return json.dumps({"objects": [obj]})
                        except:
                            continue
            
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and 'objects' in parsed:
                    valid_objects = []
                    for obj in parsed['objects']:
                        if isinstance(obj, dict) and obj.get('name') != "Object Name":
                            valid_objects.append(obj)
                    
                    if valid_objects:
                        return json.dumps({"objects": valid_objects})
                    else:
                        return '{"objects": []}'
            except:
                pass
            
            json_pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*\}'
            matches = re.findall(json_pattern, cleaned, re.DOTALL)
            
            if matches:
                objects_list = []
                for match_text in re.findall(r'\{[^{}]*"name"\s*:\s*"[^"]*"[^{}]*\}', cleaned, re.DOTALL):
                    try:
                        obj = json.loads(match_text)
                        if isinstance(obj, dict) and 'name' in obj and obj['name'] != "Object Name":
                            objects_list.append(obj)
                    except:
                        continue
                
                if objects_list:
                    return json.dumps({"objects": objects_list})
        
        return '{"objects": []}'
    
    def _clean_json_output(self, text):
        text = text.strip()
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return text
    
    def generate_query_specific_sg(self, image, question, object_list):
        try:
            readable_objects = "- No specific objects detected, analyze the image directly"
            extracted_objects = []
            
            try:
                objects_data = json.loads(object_list)
                if 'objects' in objects_data and objects_data['objects']:
                    objects = objects_data['objects']
                    object_names = []
                    
                    for obj in objects[:20]:
                        if isinstance(obj, dict):
                            name = obj.get('name', 'Unknown')
                            obj_type = obj.get('type', '')
                            description = obj.get('description', '')
                            
                            if name == "Object Name" or "Object Name" in name:
                                continue
                            
                            extracted_objects.append(obj)
                            
                            obj_desc = f"- {name}"
                            if obj_type and obj_type != "Object Type":
                                obj_desc += f" ({obj_type})"
                            if description and not description.startswith("Brief description"):
                                obj_desc += f": {description[:100]}"
                            
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
For the provided image ,its object list and the associated question, generate a scene graph in JSON format that includes the following:

1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question  
3. Object relationships that are relevant to answering the question

Available Objects in Image:
{readable_objects}



Question: "{question}"



Focus on accuracy and relevance to the question. Only include objects and relationships that help answer the specific query.

A:"""
            else:
                prompt = f"""USER: <image>
For the provided image and its associated question, generate a scene graph in JSON format that includes the following:

1. Objects that are relevant to answering the question
2. Object attributes that are relevant to answering the question  
3. Object relationships that are relevant to answering the question




Question: "{question}"



Focus on accuracy and relevance to the question. Only include objects and relationships that help answer the specific query.

A:"""
            
            inputs = self.processor(text=prompt, images=image, return_tensors="pt", padding=True)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            if "ASSISTANT:" in generated_text:
                sg_text = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                prompt_clean = prompt.replace("<image>", "").replace("USER:", "").strip()
                sg_text = generated_text.replace(prompt_clean, "").strip()
            
            sg_text = self._clean_json_output(sg_text)
            
            try:
                parsed_sg = json.loads(sg_text)
                if isinstance(parsed_sg, dict) and ('objects' in parsed_sg or 'relationships' in parsed_sg):
                    if 'objects' not in parsed_sg:
                        parsed_sg['objects'] = []
                    if 'relationships' not in parsed_sg:
                        parsed_sg['relationships'] = []
                    return json.dumps(parsed_sg)
            except Exception as e:
                print(f"Error validating generated scene graph: {e}")
            
            try:
                json_match = re.search(r'\{[^}]*"objects"[^}]*\}', sg_text, re.DOTALL)
                if json_match:
                    candidate = json_match.group(0)
                    parsed = json.loads(candidate)
                    if 'objects' not in parsed:
                        parsed['objects'] = []
                    if 'relationships' not in parsed:
                        parsed['relationships'] = []
                    return json.dumps(parsed)
            except:
                pass
            
            return '{"objects": [], "relationships": []}'
            
        except Exception as e:
            print(f"Error generating scene graph: {e}")
            return f'{{"error": "Error generating scene graph: {str(e)}"}}'
    
    def load_and_clean_results(self, filepath="results_seed.json"):
        print(f"Loading and cleaning results from {filepath}...")
        
        if not os.path.exists(filepath):
            print(f"{filepath} file not found!")
            return []
        
        results = self._parse_malformed_json_file(filepath)
        
        cleaned_results = []
        for result in results:
            if self._validate_result_object(result):
                result['object_list'] = self._extract_and_clean_object_list(
                    result.get('object_list', '')
                )
                cleaned_results.append(result)
        
        print(f"Loaded and validated {len(cleaned_results)} results")
        return cleaned_results
    
    def load_seed_images(self):
        print("Loading SEED-Bench dataset...")
        
        try:
            dataset = load_dataset("lmms-lab/SEED-Bench", trust_remote_code=True)
            test_data = dataset['test']
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return {}
        
        image_lookup = {}
        print("Building image lookup...")
        
        for sample in tqdm(test_data, desc="Indexing images"):
            try:
                data_id = str(sample.get('data_id', ''))
                if not data_id:
                    continue
                
                image = None
                if 'image' in sample and sample['image']:
                    image_data = sample['image']
                    
                    if isinstance(image_data, list) and len(image_data) > 0:
                        image = image_data[0]
                    else:
                        image = image_data
                    
                    if hasattr(image, 'convert'):
                        image_lookup[data_id] = image.convert('RGB')
                    elif image is not None:
                        image_lookup[data_id] = image
                        
            except Exception as e:
                print(f"Error processing sample {sample.get('data_id', 'unknown')}: {e}")
                continue
        
        print(f"Indexed {len(image_lookup)} images")
        return image_lookup
    
    def debug_object_extraction(self, sample_data, num_samples=5):
        print("DEBUG: Object List Extraction Results")
        print("=" * 50)
        
        for i, sample in enumerate(sample_data[:num_samples]):
            unique_id = sample.get('unique_image_id', 'Unknown')
            question = sample.get('question', 'No question')[:50] + "..."
            raw_object_list = sample.get('object_list', '')
            
            print(f"\nSample {i+1}: {unique_id}")
            print(f"Question: {question}")
            print(f"Raw object_list length: {len(raw_object_list)}")
            
            if "A:" in raw_object_list:
                parts = raw_object_list.split("A:")
                print(f"Content before A:: {parts[0][-50:]}")
                print(f"Content after A:: {parts[-1][:200]}")
            else:
                print(f"Raw object_list preview: {raw_object_list[:200]}...")
            
            extracted = self._extract_and_clean_object_list(raw_object_list)
            print(f"Extracted object_list: {extracted}")
            
            try:
                parsed = json.loads(extracted)
                if 'objects' in parsed:
                    print(f"Number of objects found: {len(parsed['objects'])}")
                    for j, obj in enumerate(parsed['objects'][:3]):
                        name = obj.get('name', 'Unknown')
                        obj_type = obj.get('type', 'Unknown type')
                        is_template = name == "Object Name" or "Object Name" in name
                        print(f"  Object {j+1}: {name} ({obj_type}) [Template: {is_template}]")
                else:
                    print("No 'objects' key found in extracted JSON")
            except Exception as e:
                print(f"Failed to parse extracted JSON: {e}")
            
            print("-" * 30)
    
    def test_specific_sample_extraction(self):
        print("Testing problematic sample extraction:")
        print("=" * 50)
        
        problematic_input = """{\n    \"objects\": [\n        {\n            \"name\": \"Object Name\",\n            \"type\": \"Object Type (e.g., Living Entity, Environmental Element, Man-made Object)\",\n            \"description\": \"Brief description of the object and its relevance to the question\"\n        }\n    ]\n}\n\nFocus on accuracy and relevance to the question. Only include objects that help answer the specific query.\n\nObjects List (in JSON format only):\n\nA:\n{\n\"name\": \"Microphone\",\n\"type\": \"Living Entity\",\n\"description\": \"A microphone is being held by one of the men in the image.\"\n}"""
        
        print("Input:")
        print(problematic_input)
        print("\nStep-by-step processing:")
        
        cleaned = problematic_input.replace('\\n', '\n').replace('\\"', '"')
        print(f"1. Cleaned input length: {len(cleaned)}")
        
        if "A:" in cleaned:
            print("2. Found 'A:' in input")
            after_a = cleaned.split("A:")[-1].strip()
            print(f"3. Content after A:: '{after_a}'")
            
            start = after_a.find('{')
            print(f"4. JSON start position: {start}")
            
            if start != -1:
                brace_count = 0
                end = len(after_a)
                
                for i, char in enumerate(after_a[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                
                json_part = after_a[start:end]
                print(f"5. Extracted JSON part: '{json_part}'")
                
                try:
                    obj = json.loads(json_part)
                    print(f"6. Parsed object: {obj}")
                    
                    if isinstance(obj, dict) and 'name' in obj:
                        result = json.dumps({"objects": [obj]})
                        print(f"7. Final result: {result}")
                        return result
                except Exception as e:
                    print(f"6. JSON parsing error: {e}")
        
        print("\nTesting actual extraction function:")
        result = self._extract_and_clean_object_list(problematic_input)
        print(f"Function result: {result}")
        
        return result
    
    def process_query_specific_sg(self, results_file="results_seed.json", max_samples=None, debug=False):
        results = self.load_and_clean_results(results_file)
        
        if not results:
            print("No valid results found! Please check your results file.")
            return []
        
        print(f"Found {len(results)} samples to process")
        
        if debug:
            self.debug_object_extraction(results)
            response = input("\nContinue with processing? (y/n): ")
            if response.lower() != 'y':
                return []
        
        if max_samples:
            results = results[:max_samples]
            print(f"Limited to {len(results)} samples")
        
        image_lookup = self.load_seed_images()
        
        if not image_lookup:
            print("No images loaded from SEED dataset!")
            return []
        
        query_sg_results = []
        missing_images = 0
        processing_errors = 0
        
        print("Generating query-specific scene graphs...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"query_sg_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        for i, result in enumerate(tqdm(results, desc="Processing samples")):
            try:
                unique_image_id = str(result['unique_image_id'])
                question = result['question']
                question_id = str(result.get('question_id', ''))
                object_list = result.get('object_list', '{"objects": []}')
                
                if unique_image_id not in image_lookup:
                    print(f"Image {unique_image_id} not found in dataset")
                    missing_images += 1
                    continue
                
                image = image_lookup[unique_image_id]
                
                query_sg = self.generate_query_specific_sg(image, question, object_list)
                
                sg_result = {
                    'unique_image_id': unique_image_id,
                    'question_id': question_id,
                    'question': question,
                    'object_list': object_list,
                    'scene_graph': query_sg,
                    'timestamp': datetime.now().isoformat()
                }
                
                query_sg_results.append(sg_result)
                
                if (i + 1) % 50 == 0:
                    intermediate_file = os.path.join(results_dir, f'sg_intermediate_{i+1}.json')
                    with open(intermediate_file, 'w') as f:
                        json.dump(query_sg_results, f, indent=2)
                    print(f"Saved intermediate results ({i+1} processed)")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                processing_errors += 1
                continue
        
        final_results_file = os.path.join(results_dir, 'sg_final.json')
        with open(final_results_file, 'w') as f:
            json.dump(query_sg_results, f, indent=2)
        
        with open('sg.json', 'w') as f:
            json.dump(query_sg_results, f, indent=2)
        
        summary = {
            'timestamp': timestamp,
            'total_input_samples': len(results),
            'successful_generations': len(query_sg_results),
            'missing_images': missing_images,
            'processing_errors': processing_errors,
            'success_rate': len(query_sg_results) / len(results) if results else 0,
            'results_directory': results_dir
        }
        
        summary_file = os.path.join(results_dir, 'processing_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Successfully generated {len(query_sg_results)} query-specific scene graphs!")
        print(f"Missing images: {missing_images}")
        print(f"Processing errors: {processing_errors}")
        print(f"Results saved to: {final_results_file}")
        print(f"Also saved to: sg.json")
        
        return query_sg_results

def main():
    print("Query-Specific Scene Graph Generation")
    print("=" * 50)
    
    results_file = "results_seed.json"
    max_samples = None
    debug_mode = True
    test_extraction = True
    
    print(f"Configuration:")
    print(f"  Results file: {results_file}")
    print(f"  Max samples: {max_samples or 'All'}")
    print(f"  Debug mode: {debug_mode}")
    print(f"  Test extraction: {test_extraction}")
    
    try:
        generator = ImprovedQuerySpecificSGGenerator()
        
        if test_extraction:
            print("\n" + "="*50)
            generator.test_specific_sample_extraction()
            print("="*50)
            
            response = input("\nContinue with full processing? (y/n): ")
            if response.lower() != 'y':
                return
        
        results = generator.process_query_specific_sg(
            results_file=results_file,
            max_samples=max_samples,
            debug=debug_mode
        )
        
        print(f"\nProcessing complete!")
        print(f"Total scene graphs generated: {len(results)}")
        
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
            template_count = 0
            
            for result in results:
                try:
                    obj_list = json.loads(result['object_list'])
                    has_templates = False
                    has_valid_objects = False
                    
                    if 'objects' in obj_list:
                        for obj in obj_list['objects']:
                            if obj.get('name') == 'Object Name':
                                has_templates = True
                            else:
                                has_valid_objects = True
                    
                    sg = json.loads(result['scene_graph'])
                    if 'objects' in sg and sg['objects']:
                        valid_count += 1
                    else:
                        empty_count += 1
                        if has_templates and not has_valid_objects:
                            template_count += 1
                            
                except:
                    empty_count += 1
            
            print(f"\nScene Graph Quality:")
            print(f"  Valid scene graphs (with objects): {valid_count}")
            print(f"  Empty scene graphs: {empty_count}")
            print(f"  Empty due to templates only: {template_count}")
            print(f"  Success rate: {valid_count/len(results)*100:.1f}%")
            
            if template_count > len(results) * 0.5:
                print("\nWARNING: High number of template-only samples detected!")
                print("This indicates the object extraction from 'A:' sections is not working properly.")
            
            if empty_count > len(results) * 0.8:
                print("\nWARNING: High number of empty scene graphs detected!")
                print("This might indicate:")
                print("  - Issues with object list extraction")
                print("  - Problems with model generation")
                print("  - Insufficient prompt clarity")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()