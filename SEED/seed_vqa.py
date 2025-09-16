import torch
import os
from tqdm import tqdm
import json
from datetime import datetime
from PIL import Image
import re
import gc
from datasets import load_dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration

class SEEDVQAEvaluator:
    def __init__(self, 
                 global_scene_graphs_path="results_seed.json",
                 query_scene_graphs_path="/teamspace/studios/this_studio/query_sg_results_20250909_234620/sg_final.json"):
        
        print("Loading LLaVA model for VQA...")
        self.processor = None
        self.model = None
        self._load_model()
        
        print(f"Loading global scene graphs from {global_scene_graphs_path}...")
        self.global_lookup = {}
        self._load_global_scene_graphs(global_scene_graphs_path)
        
        print(f"Loading query-specific scene graphs from {query_scene_graphs_path}...")
        self.query_lookup = {}
        self._load_query_scene_graphs(query_scene_graphs_path)
        
        print("Loading SEED-Bench dataset...")
        self.dataset = None
        self.data_lookup = {}
        self._load_seed_dataset()
    
    def _load_model(self):
        try:
            model_name = "llava-hf/llava-1.5-7b-hf"
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            print("LLaVA model loaded successfully")
            
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
            
            lines = content.split('\n')
            current_obj = ""
            brace_count = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if not ('{' in line or current_obj):
                    continue
                
                current_obj += line + " "
                
                brace_count += line.count('{') - line.count('}')
                
                if brace_count == 0 and current_obj.strip():
                    obj_str = current_obj.strip().rstrip(',')
                    parsed = self._safe_json_loads(obj_str)
                    
                    if parsed:
                        objects.append(parsed)
                    else:
                        extracted = self._extract_json_from_text(obj_str)
                        if extracted:
                            objects.append(extracted)
                    
                    current_obj = ""
                    brace_count = 0
            
            print(f"Parsed {len(objects)} objects from malformed JSON")
            return objects
            
        except Exception as e:
            print(f"Error parsing file {filepath}: {e}")
            return []
    
    def _load_global_scene_graphs(self, path):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        
        try:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    global_data = json.load(f)
                print("Loaded JSON normally")
            except json.JSONDecodeError:
                print("JSON file appears malformed, attempting manual parsing...")
                global_data = self._parse_malformed_json_file(path)
            
            if not global_data:
                print("No data loaded from global scene graphs file")
                return
            
            print(f"Processing {len(global_data)} global scene graph items...")
            success_count = 0
            
            for i, item in enumerate(global_data):
                try:
                    unique_image_id = str(item.get('unique_image_id', ''))
                    question_id = str(item.get('question_id', ''))
                    
                    if not unique_image_id or not question_id:
                        continue
                    
                    composite_key = f"{unique_image_id}_{question_id}"
                    
                    global_sg_raw = item.get('global_scene_graph', '')
                    
                    scene_graph = self._extract_and_validate_scene_graph(global_sg_raw)
                    if not scene_graph:
                        object_list = item.get('object_list', '')
                        scene_graph = self._extract_and_validate_scene_graph(object_list)
                        if not scene_graph:
                            scene_graph = {"objects": [{"name": "Unknown", "attributes": []}], "relationships": []}
                    
                    self.global_lookup[composite_key] = {
                        'data_id': unique_image_id,
                        'question_id': question_id,
                        'question': item.get('question', ''),
                        'global_scene_graph': json.dumps(scene_graph),
                        'raw_data': item
                    }
                    success_count += 1
                    
                except Exception as e:
                    print(f"Error processing global item {i}: {e}")
                    continue
            
            print(f"Loaded global scene graphs for {success_count} image-question pairs")
            
        except Exception as e:
            print(f"Error loading global scene graphs: {e}")
    
    def _extract_and_validate_scene_graph(self, raw_data):
        if not raw_data:
            return None
        
        if isinstance(raw_data, dict):
            if 'objects' in raw_data:
                return raw_data
            elif 'scene_graph' in raw_data and 'objects' in raw_data['scene_graph']:
                return raw_data['scene_graph']
        
        if isinstance(raw_data, str):
            cleaned = raw_data.replace('\\_', '_').strip()
            
            if "A:" in cleaned:
                after_a = cleaned.split("A:")[-1].strip()
                extracted = self._extract_json_from_text(after_a)
                if extracted:
                    if 'scene_graph' in extracted:
                        return extracted['scene_graph']
                    elif 'objects' in extracted:
                        return extracted
            
            parsed = self._safe_json_loads(cleaned)
            if parsed:
                if 'scene_graph' in parsed:
                    return parsed['scene_graph']
                elif 'objects' in parsed:
                    return parsed
            
            extracted = self._extract_json_from_text(cleaned)
            if extracted:
                return extracted
        
        return None
    
    def _load_query_scene_graphs(self, path):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                query_data = json.load(f)
            
            print(f"Processing {len(query_data)} query scene graph items...")
            success_count = 0
            
            for i, item in enumerate(query_data):
                try:
                    unique_image_id = str(item.get('unique_image_id', ''))
                    question_id = str(item.get('question_id', ''))
                    
                    if not unique_image_id or not question_id:
                        continue
                    
                    composite_key = f"{unique_image_id}_{question_id}"
                    
                    object_list_raw = item.get('object_list', '')
                    scene_graph = self._extract_and_validate_scene_graph(object_list_raw)
                    
                    if not scene_graph:
                        scene_graph = {"objects": [{"name": "Unknown", "attributes": []}], "relationships": []}
                    
                    self.query_lookup[composite_key] = {
                        'data_id': unique_image_id,
                        'question_id': question_id,
                        'question': item.get('question', ''),
                        'query_scene_graph': json.dumps(scene_graph),
                        'raw_data': item
                    }
                    success_count += 1
                    
                except Exception as e:
                    print(f"Error processing query item {i}: {e}")
                    continue
            
            print(f"Loaded query-specific scene graphs for {success_count} image-question pairs")
            
        except Exception as e:
            print(f"Error loading query scene graphs: {e}")
    
    def _load_seed_dataset(self):
        try:
            print("Downloading SEED-Bench dataset...")
            
            Image.MAX_IMAGE_PIXELS = None
            
            dataset = load_dataset("lmms-lab/SEED-Bench", trust_remote_code=True)
            self.dataset = dataset['test']
            
            print("Building dataset lookup...")
            
            chunk_size = 1000
            total_samples = len(self.dataset)
            
            for i in tqdm(range(0, total_samples, chunk_size), desc="Indexing dataset (chunked)"):
                chunk_end = min(i + chunk_size, total_samples)
                
                for idx in range(i, chunk_end):
                    try:
                        sample = self.dataset[idx]
                        data_id = str(sample.get('data_id', ''))
                        question_id = str(sample.get('question_id', ''))
                        
                        if data_id and question_id:
                            composite_key = f"{data_id}_{question_id}"
                            
                            self.data_lookup[composite_key] = {
                                'data_id': data_id,
                                'question_id': question_id,
                                'question': sample.get('question', ''),
                                'choice_a': sample.get('choice_a', ''),
                                'choice_b': sample.get('choice_b', ''),
                                'choice_c': sample.get('choice_c', ''),
                                'choice_d': sample.get('choice_d', ''),
                                'answer': sample.get('answer', ''),
                                'image_index': idx
                            }
                    except Exception as e:
                        print(f"Error processing sample {idx}: {e}")
                        continue
                
                if i % (chunk_size * 5) == 0:
                    gc.collect()
                    print(f"  Processed {min(i + chunk_size, total_samples)}/{total_samples} samples, "
                          f"Memory cleanup performed")
            
            print(f"Loaded {len(self.data_lookup)} samples from SEED dataset")
            
            gc.collect()
            
        except Exception as e:
            print(f"Error loading SEED dataset: {e}")
            self.dataset = None
            self.data_lookup = {}
    
    def get_sample_data(self, composite_key):
        if composite_key not in self.data_lookup:
            return None
        
        sample_info = self.data_lookup[composite_key]
        
        try:
            image_index = sample_info['image_index']
            full_sample = self.dataset[image_index]
            
            image = None
            if 'image' in full_sample and full_sample['image']:
                image_data = full_sample['image']
                if isinstance(image_data, list) and len(image_data) > 0:
                    image = image_data[0]
                else:
                    image = image_data
                
                if hasattr(image, 'convert'):
                    image = image.convert('RGB')
                    if hasattr(image, 'size'):
                        width, height = image.size
                        max_size = 1024
                        if width > max_size or height > max_size:
                            ratio = min(max_size/width, max_size/height)
                            new_width = int(width * ratio)
                            new_height = int(height * ratio)
                            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            return {
                'image': image,
                'question': sample_info['question'],
                'choice_a': sample_info['choice_a'],
                'choice_b': sample_info['choice_b'],
                'choice_c': sample_info['choice_c'],
                'choice_d': sample_info['choice_d'],
                'answer': sample_info['answer'],
                'question_id': sample_info['question_id'],
                'data_id': sample_info['data_id']
            }
            
        except Exception as e:
            print(f"Error retrieving sample data for {composite_key}: {e}")
            return None
    
    def create_mcq_prompt(self, global_sg, query_sg, question, choices):
        choice_text = f"""
A. {choices['choice_a']}
B. {choices['choice_b']}
C. {choices['choice_c']}
D. {choices['choice_d']}
"""
        
        prompt = f"""USER: <image>

Global Scene Graph: {global_sg}
Query-Specific Scene Graph: {query_sg}
Question: {question}

Choices:{choice_text}

Instructions : Analyze the image and both the scene graphs to
answer the multiple choice question . The global scene graph
gives the overall context of the image and the query specific
scene graph gives context relevant for answering the question .
Answer with only the letter (A , B , C , or D ) of the correct
choice which is the most appropriate .
A:"""
        
        return prompt
    
    def generate_mcq_response(self, image, global_sg, query_sg, question, choices):
        try:
            prompt = self.create_mcq_prompt(global_sg, query_sg, question, choices)
            
            inputs = self.processor(
                text=prompt, 
                images=image, 
                return_tensors="pt",
                padding=True
            )
            
            if torch.cuda.is_available():
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            if "ASSISTANT:" in generated_text:
                response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                prompt_without_image = prompt.replace("<image>", "").replace("USER:", "").strip()
                response = generated_text.replace(prompt_without_image, "").strip()
            
            response = response.strip().upper()
            
            for char in ['A', 'B', 'C', 'D']:
                if char in response:
                    return char
            
            return 'A'
            
        except Exception as e:
            print(f"Error generating MCQ response: {e}")
            return 'A'
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def get_matching_samples(self):
        print(f"Finding matching samples...")
        print(f"  Global lookup: {len(self.global_lookup)} items")
        print(f"  Query lookup: {len(self.query_lookup)} items")  
        print(f"  Dataset lookup: {len(self.data_lookup)} items")
        
        matching_samples = []
        
        global_keys = set(self.global_lookup.keys())
        query_keys = set(self.query_lookup.keys())
        dataset_keys = set(self.data_lookup.keys())
        
        common_global_query = global_keys & query_keys
        final_common = common_global_query & dataset_keys
        
        print(f"  Common between global and query: {len(common_global_query)}")
        print(f"  Final matching (all three sources): {len(final_common)}")
        
        if len(final_common) == 0:
            print(f"\nDebugging - Sample keys:")
            print(f"  Global keys (first 5): {list(global_keys)[:5]}")
            print(f"  Query keys (first 5): {list(query_keys)[:5]}")
            print(f"  Dataset keys (first 5): {list(dataset_keys)[:5]}")
        
        for composite_key in final_common:
            global_data = self.global_lookup[composite_key]
            query_data = self.query_lookup[composite_key]
            
            matching_samples.append({
                'composite_key': composite_key,
                'data_id': global_data.get('data_id', ''),
                'question_id': global_data['question_id'],
                'global_scene_graph': global_data['global_scene_graph'],
                'query_scene_graph': query_data['query_scene_graph']
            })
        
        print(f"Final matching samples: {len(matching_samples)}")
        
        if matching_samples:
            sample_keys = [s['composite_key'] for s in matching_samples[:5]]
            print(f"  Sample matching keys: {sample_keys}")
        
        return matching_samples
    
    def evaluate_dataset(self, max_samples=None):
        print("Starting SEED-Bench MCQ Evaluation with Scene Graphs")
        print("="*60)
        
        results = []
        correct_count = 0
        total_count = 0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"seed_mcq_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        matching_samples = self.get_matching_samples()
        
        if len(matching_samples) == 0:
            print("ERROR: No matching samples found!")
            print("This could be due to:")
            print("  - Mismatched composite key formats between files")
            print("  - Issues with scene graph parsing")
            print("  - Dataset loading problems")
            return None
        
        samples_to_process = matching_samples[:max_samples] if max_samples else matching_samples
        print(f"Processing {len(samples_to_process)} samples...")
        
        for i, sample_data in enumerate(tqdm(samples_to_process, desc="MCQ Evaluation")):
            composite_key = sample_data['composite_key']
            data_id = sample_data['data_id']
            question_id = sample_data['question_id']
            global_sg = sample_data['global_scene_graph']
            query_sg = sample_data['query_scene_graph']
            
            dataset_sample = self.get_sample_data(composite_key)
            if not dataset_sample or not dataset_sample['image']:
                print(f"  Warning: Could not get data for {composite_key}, skipping...")
                continue
            
            try:
                image = dataset_sample['image']
                
                dataset_question = dataset_sample['question']
                choices = {
                    'choice_a': dataset_sample['choice_a'],
                    'choice_b': dataset_sample['choice_b'],
                    'choice_c': dataset_sample['choice_c'],
                    'choice_d': dataset_sample['choice_d']
                }
                ground_truth = dataset_sample['answer']
                
                predicted_answer = self.generate_mcq_response(image, global_sg, query_sg, dataset_question, choices)
                
                is_correct = predicted_answer == ground_truth
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                result = {
                    'composite_key': composite_key,
                    'data_id': data_id,
                    'question_id': question_id,
                    'question': dataset_question,
                    'choice_a': choices['choice_a'],
                    'choice_b': choices['choice_b'],
                    'choice_c': choices['choice_c'],
                    'choice_d': choices['choice_d'],
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'global_scene_graph_preview': global_sg[:200] + "..." if len(global_sg) > 200 else global_sg,
                    'query_scene_graph_preview': query_sg[:200] + "..." if len(query_sg) > 200 else query_sg
                }
                
                results.append(result)
                
                if total_count % 10 == 0:
                    current_accuracy = correct_count / total_count
                    print(f"   Processed {total_count}/{len(samples_to_process)} samples, Accuracy: {current_accuracy:.4f}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                if total_count % 50 == 0:
                    intermediate_file = os.path.join(results_dir, f'intermediate_results_{total_count}.json')
                    with open(intermediate_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
            
            except Exception as e:
                print(f"Error processing sample {composite_key}: {e}")
                continue
        
        final_accuracy = correct_count / total_count if total_count > 0 else 0
        
        detailed_file = os.path.join(results_dir, 'detailed_results.json')
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        summary = {
            'timestamp': timestamp,
            'dataset': 'SEED-Bench',
            'evaluation_type': 'Multiple Choice VQA with Scene Graphs',
            'total_samples': total_count,
            'correct_answers': correct_count,
            'accuracy': final_accuracy,
            'methodology': 'MCQ VQA using global and query-specific scene graphs with images, matched by composite key (data_id + question_id)',
            'config': {
                'max_samples': max_samples,
                'model': 'llava-hf/llava-1.5-13b-hf',
                'global_scene_graphs_loaded': len(self.global_lookup),
                'query_scene_graphs_loaded': len(self.query_lookup),
                'dataset_samples_loaded': len(self.data_lookup),
                'matching_samples_found': len(matching_samples),
                'matching_strategy': 'composite_key_data_id_question_id'
            }
        }
        
        summary_file = os.path.join(results_dir, 'evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nSEED-Bench MCQ evaluation completed!")
        print(f"Results Summary:")
        print(f"   Total Samples: {total_count}")
        print(f"   Correct Answers: {correct_count}")
        print(f"   Final Accuracy: {final_accuracy:.4f}")
        print(f"   Results saved to: {results_dir}")
        
        return {
            'accuracy': final_accuracy,
            'results': results,
            'summary': summary,
            'results_dir': results_dir
        }

def main():
    print("SEED-Bench Multiple Choice VQA Evaluation (Memory Optimized)")
    print("="*50)
    
    global_file = "results_seed.json"
    query_file = "/teamspace/studios/this_studio/query_sg_results_20250909_234620/sg_final.json"
    max_samples = 1000
    
    print(f"Configuration:")
    print(f"  Global scene graphs: {global_file}")
    print(f"  Query scene graphs: {query_file}")
    print(f"  Max samples: {max_samples}")
    print(f"  Matching strategy: Composite key (unique_image_id/data_id + question_id)")
    
    if not os.path.exists(global_file):
        print(f"ERROR: {global_file} not found!")
        return
    
    if not os.path.exists(query_file):
        print(f"ERROR: {query_file} not found!")
        return
    
    try:
        evaluator = SEEDVQAEvaluator(
            global_scene_graphs_path=global_file,
            query_scene_graphs_path=query_file
        )
    except Exception as e:
        print(f"ERROR initializing evaluator: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\nData Summary:")
    print(f"  Global scene graphs: {len(evaluator.global_lookup)}")
    print(f"  Query scene graphs: {len(evaluator.query_lookup)}")
    print(f"  Dataset samples: {len(evaluator.data_lookup)}")
    
    if len(evaluator.global_lookup) == 0:
        print("ERROR: No global scene graphs loaded!")
        return
    
    if len(evaluator.query_lookup) == 0:
        print("ERROR: No query scene graphs loaded!")
        return
    
    if len(evaluator.data_lookup) == 0:
        print("ERROR: No dataset samples loaded!")
        return
    
    matching_samples = evaluator.get_matching_samples()
    
    if len(matching_samples) == 0:
        print("ERROR: No matching samples found!")
        print("\nDebugging information:")
        print("The composite key format should be: data_id_question_id")
        print("Check that your scene graph files have both unique_image_id and question_id fields")
        return
    
    try:
        results = evaluator.evaluate_dataset(max_samples=max_samples)
        
        if results:
            print(f"\nEvaluation Summary:")
            print(f"  Final Accuracy: {results['accuracy']:.4f}")
            print(f"  Results Directory: {results['results_dir']}")
            
            if results['results']:
                print(f"\nSample Results:")
                for i, result in enumerate(results['results'][:3]):
                    status = "CORRECT" if result['is_correct'] else "WRONG"
                    print(f"  {status} Sample {i+1} (Key: {result['composite_key']}):")
                    print(f"    Question: {result['question'][:60]}...")
                    print(f"    Ground Truth: {result['ground_truth']}")
                    print(f"    Predicted: {result['predicted_answer']}")
        
    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()