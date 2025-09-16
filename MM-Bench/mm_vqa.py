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

class MMBenchVQAEvaluator:
    """VQA evaluator for MMBench multiple choice questions using scene graphs"""
    
    def __init__(self, 
                 global_scene_graphs_path="results_mmbench.json",
                 query_scene_graphs_path="mmbench_query_sg_results_20250913_120359/mmbench_sg_final.json"):
        
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
        
        print("Loading MMBench dataset...")
        self.dataset = None
        self.data_lookup = {}
        self._load_mmbench_dataset()
    
    def _load_model(self):
        """Load LLaVA model with memory optimizations"""
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
    
    def _extract_json_from_text(self, text):
        """Extract JSON object from text"""
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
        """Parse malformed JSON file line by line"""
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

    def _extract_choices_from_item(self, item):
        """Extract choices from item data, handling MMBench JSON format"""
        choices = {
            'choice_a': '',
            'choice_b': '',
            'choice_c': '',
            'choice_d': ''
        }
        
        choice_fields = ['choice_a', 'choice_b', 'choice_c', 'choice_d']
        for field in choice_fields:
            if field in item and item[field] not in ['nan', '', None]:
                choices[field] = str(item[field])
        
        if 'options' in item and item['options']:
            options = item['options']
            
            if isinstance(options, dict):
                option_mapping = {
                    'A': 'choice_a',
                    'B': 'choice_b', 
                    'C': 'choice_c',
                    'D': 'choice_d'
                }
                
                for option_key, choice_key in option_mapping.items():
                    if option_key in options:
                        value = options[option_key]
                        if value not in ['nan', '', None, 'None'] and str(value).strip():
                            choices[choice_key] = str(value).strip()
            
            elif isinstance(options, list) and len(options) >= 2:
                choice_keys = ['choice_a', 'choice_b', 'choice_c', 'choice_d']
                for i, option in enumerate(options[:4]):
                    if option not in ['nan', '', None, 'None'] and str(option).strip():
                        choices[choice_keys[i]] = str(option).strip()
        
        return choices
    
    def _load_global_scene_graphs(self, path):
        """Load global scene graphs from results_mmbench.json"""
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
                    original_dataset_index = str(item.get('original_dataset_index', ''))
                    
                    if not original_dataset_index:
                        continue
                    
                    global_sg_raw = item.get('global_scene_graph', '')
                    
                    scene_graph = self._extract_and_validate_scene_graph(global_sg_raw)
                    if not scene_graph:
                        object_list = item.get('object_list', '')
                        scene_graph = self._extract_and_validate_scene_graph(object_list)
                        if not scene_graph:
                            scene_graph = {"objects": [{"name": "Unknown", "attributes": []}], "relationships": []}
                    
                    choices = self._extract_choices_from_item(item)
                    
                    self.global_lookup[original_dataset_index] = {
                        'original_dataset_index': original_dataset_index,
                        'question': item.get('question', ''),
                        'category': item.get('category', ''),
                        'l2_category': item.get('l2_category', ''),
                        'answer': item.get('answer', ''),
                        'choice_a': choices.get('choice_a', ''),
                        'choice_b': choices.get('choice_b', ''),
                        'choice_c': choices.get('choice_c', ''),
                        'choice_d': choices.get('choice_d', ''),
                        'options': item.get('options', {}),
                        'global_scene_graph': json.dumps(scene_graph),
                        'raw_data': item
                    }
                    success_count += 1
                    
                except Exception as e:
                    print(f"Error processing global item {i}: {e}")
                    continue
            
            print(f"Loaded global scene graphs for {success_count} samples")
            
        except Exception as e:
            print(f"Error loading global scene graphs: {e}")
    
    def _extract_and_validate_scene_graph(self, raw_data):
        """Extract and validate scene graph from raw data"""
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
        """Load query-specific scene graphs from mmbench_sg_final.json"""
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
                    original_dataset_index = str(item.get('original_dataset_index', ''))
                    
                    if not original_dataset_index:
                        continue
                    
                    query_sg_raw = item.get('query_specific_scene_graph', '')
                    scene_graph = self._extract_and_validate_scene_graph(query_sg_raw)
                    
                    if not scene_graph:
                        scene_graph = {"objects": [{"name": "Unknown", "attributes": []}], "relationships": []}
                    
                    choices = self._extract_choices_from_item(item)
                    
                    self.query_lookup[original_dataset_index] = {
                        'original_dataset_index': original_dataset_index,
                        'question': item.get('question', ''),
                        'category': item.get('category', ''),
                        'l2_category': item.get('l2_category', ''),
                        'answer': item.get('answer', ''),
                        'choice_a': choices.get('choice_a', ''),
                        'choice_b': choices.get('choice_b', ''),
                        'choice_c': choices.get('choice_c', ''),
                        'choice_d': choices.get('choice_d', ''),
                        'options': item.get('options', {}),
                        'query_scene_graph': json.dumps(scene_graph),
                        'raw_data': item
                    }
                    success_count += 1
                    
                except Exception as e:
                    print(f"Error processing query item {i}: {e}")
                    continue
            
            print(f"Loaded query-specific scene graphs for {success_count} samples")
            
        except Exception as e:
            print(f"Error loading query scene graphs: {e}")
    
    def _load_mmbench_dataset(self):
        """Load MMBench dataset and build lookups with memory optimizations"""
        try:
            print("Downloading MMBench dataset...")
            
            Image.MAX_IMAGE_PIXELS = None
            
            dataset = load_dataset("lmms-lab/MMBench_EN")
            self.dataset = dataset['dev']
            
            print("Building dataset lookup...")
            
            chunk_size = 1000
            total_samples = len(self.dataset)
            
            for i in tqdm(range(0, total_samples, chunk_size), desc="Indexing dataset (chunked)"):
                chunk_end = min(i + chunk_size, total_samples)
                
                for idx in range(i, chunk_end):
                    try:
                        sample = self.dataset[idx]
                        
                        original_dataset_index = str(idx)
                        
                        self.data_lookup[original_dataset_index] = {
                            'original_dataset_index': original_dataset_index,
                            'image_index': idx
                        }
                        
                    except Exception as e:
                        print(f"Error processing sample {idx}: {e}")
                        continue
                
                if i % (chunk_size * 5) == 0:
                    gc.collect()
                    print(f"  Processed {min(i + chunk_size, total_samples)}/{total_samples} samples, "
                          f"Memory cleanup performed")
            
            print(f"Loaded {len(self.data_lookup)} samples from MMBench dataset")
            
            gc.collect()
            
        except Exception as e:
            print(f"Error loading MMBench dataset: {e}")
            self.dataset = None
            self.data_lookup = {}
    
    def get_sample_data(self, original_dataset_index):
        """Get sample data including image from MMBench dataset"""
        if original_dataset_index not in self.data_lookup:
            return None
        
        sample_info = self.data_lookup[original_dataset_index]
        
        try:
            image_index = sample_info['image_index']
            full_sample = self.dataset[image_index]
            
            image = None
            if 'image' in full_sample and full_sample['image']:
                image_data = full_sample['image']
                
                if hasattr(image_data, 'convert'):
                    image = image_data.convert('RGB')
                    if hasattr(image, 'size'):
                        width, height = image.size
                        max_size = 1024
                        if width > max_size or height > max_size:
                            ratio = min(max_size/width, max_size/height)
                            new_width = int(width * ratio)
                            new_height = int(height * ratio)
                            image = image.resize((new_width, new_height), Image.LANCZOS)
                else:
                    image = image_data
            
            return {
                'image': image,
                'original_dataset_index': original_dataset_index
            }
            
        except Exception as e:
            print(f"Error retrieving sample data for {original_dataset_index}: {e}")
            return None
    
    def create_mcq_prompt(self, global_sg, query_sg, question, choices):
        """Create multiple choice prompt using both scene graphs"""
        
        valid_choices = []
        choice_labels = ['A', 'B', 'C', 'D']
        choice_keys = ['choice_a', 'choice_b', 'choice_c', 'choice_d']
        
        for i, (label, key) in enumerate(zip(choice_labels, choice_keys)):
            choice_text = choices.get(key, '').strip()
            if choice_text and choice_text not in ['nan', 'None', '']:
                valid_choices.append(f"{label}. {choice_text}")
            else:
                valid_choices.append(f"{label}. [Option {label}]")
        
        choice_text = "\n" + "\n".join(valid_choices) + "\n"
        
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
        """Generate multiple choice response using both scene graphs with actual image"""
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
        """Get samples that have both scene graphs and exist in dataset"""
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
        
        for original_dataset_index in final_common:
            global_data = self.global_lookup[original_dataset_index]
            query_data = self.query_lookup[original_dataset_index]
            
            matching_samples.append({
                'original_dataset_index': original_dataset_index,
                'question': global_data['question'],
                'category': global_data.get('category', ''),
                'l2_category': global_data.get('l2_category', ''),
                'answer': global_data.get('answer', ''),
                'choice_a': global_data.get('choice_a', ''),
                'choice_b': global_data.get('choice_b', ''),
                'choice_c': global_data.get('choice_c', ''),
                'choice_d': global_data.get('choice_d', ''),
                'options': global_data.get('options', {}),
                'global_scene_graph': global_data['global_scene_graph'],
                'query_scene_graph': query_data['query_scene_graph']
            })
        
        print(f"Final matching samples: {len(matching_samples)}")
        
        if matching_samples:
            sample_keys = [s['original_dataset_index'] for s in matching_samples[:5]]
            print(f"  Sample matching keys: {sample_keys}")
            
            first_sample = matching_samples[0]
            print(f"  Sample choices for key {first_sample['original_dataset_index']}:")
            print(f"    A: {first_sample['choice_a']}")
            print(f"    B: {first_sample['choice_b']}")
            print(f"    C: {first_sample['choice_c']}")
            print(f"    D: {first_sample['choice_d']}")
        
        return matching_samples
    
    def evaluate_dataset(self, max_samples=None):
        """Evaluate multiple choice VQA using both scene graphs"""
        print("Starting MMBench MCQ Evaluation with Scene Graphs")
        print("="*60)
        
        results = []
        correct_count = 0
        total_count = 0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"mmbench_mcq_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        matching_samples = self.get_matching_samples()
        
        if len(matching_samples) == 0:
            print("ERROR: No matching samples found!")
            print("This could be due to:")
            print("  - Mismatched original_dataset_index formats between files")
            print("  - Issues with scene graph parsing")
            print("  - Dataset loading problems")
            return None
        
        samples_to_process = matching_samples[:max_samples] if max_samples else matching_samples
        print(f"Processing {len(samples_to_process)} samples...")
        
        category_stats = {}
        
        for i, sample_data in enumerate(tqdm(samples_to_process, desc="MCQ Evaluation")):
            original_dataset_index = sample_data['original_dataset_index']
            global_sg = sample_data['global_scene_graph']
            query_sg = sample_data['query_scene_graph']
            category = sample_data.get('category', 'unknown')
            l2_category = sample_data.get('l2_category', 'unknown')
            
            dataset_sample = self.get_sample_data(original_dataset_index)
            if not dataset_sample or not dataset_sample['image']:
                print(f"Warning: Could not get image for {original_dataset_index}, skipping...")
                continue
            
            try:
                image = dataset_sample['image']
                
                question = sample_data['question']
                choices = {
                    'choice_a': sample_data['choice_a'],
                    'choice_b': sample_data['choice_b'],
                    'choice_c': sample_data['choice_c'],
                    'choice_d': sample_data['choice_d']
                }
                ground_truth = sample_data['answer']
                
                predicted_answer = self.generate_mcq_response(image, global_sg, query_sg, question, choices)
                
                is_correct = predicted_answer == ground_truth
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                if category not in category_stats:
                    category_stats[category] = {'correct': 0, 'total': 0}
                category_stats[category]['total'] += 1
                if is_correct:
                    category_stats[category]['correct'] += 1
                
                result = {
                    'original_dataset_index': original_dataset_index,
                    'question': question,
                    'category': category,
                    'l2_category': l2_category,
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
                print(f"Error processing sample {original_dataset_index}: {e}")
                continue
        
        final_accuracy = correct_count / total_count if total_count > 0 else 0
        
        category_accuracies = {}
        for cat, stats in category_stats.items():
            category_accuracies[cat] = {
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                'correct': stats['correct'],
                'total': stats['total']
            }
        
        detailed_file = os.path.join(results_dir, 'detailed_results.json')
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        summary = {
            'timestamp': timestamp,
            'dataset': 'MMBench_EN',
            'evaluation_type': 'Multiple Choice VQA with Scene Graphs',
            'total_samples': total_count,
            'correct_answers': correct_count,
            'accuracy': final_accuracy,
            'category_accuracies': category_accuracies,
            'methodology': 'MCQ VQA using global and query-specific scene graphs with images, choices from scene graph files',
            'config': {
                'max_samples': max_samples,
                'model': 'llava-hf/llava-1.5-13b-hf',
                'global_scene_graphs_loaded': len(self.global_lookup),
                'query_scene_graphs_loaded': len(self.query_lookup),
                'dataset_samples_loaded': len(self.data_lookup),
                'matching_samples_found': len(matching_samples),
                'matching_strategy': 'original_dataset_index',
                'choices_source': 'scene_graph_files'
            }
        }
        
        summary_file = os.path.join(results_dir, 'evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nMMBench MCQ evaluation completed!")
        print(f"Results Summary:")
        print(f"   Total Samples: {total_count}")
        print(f"   Correct Answers: {correct_count}")
        print(f"   Final Accuracy: {final_accuracy:.4f}")
        print(f"   Results saved to: {results_dir}")
        
        print(f"\nCategory-wise Accuracy:")
        for cat, acc_data in sorted(category_accuracies.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"   {cat}: {acc_data['accuracy']:.4f} ({acc_data['correct']}/{acc_data['total']})")
        
        return {
            'accuracy': final_accuracy,
            'results': results,
            'summary': summary,
            'results_dir': results_dir,
            'category_accuracies': category_accuracies
        }

def main():
    """Main execution function"""
    print("MMBench Multiple Choice VQA Evaluation (Fixed Choices)")
    print("="*50)
    
    global_file = "results_mmbench.json"
    query_file = "mmbench_query_sg_results_20250913_120359/mmbench_sg_final.json"
    max_samples = 1000
    
    print(f"Configuration:")
    print(f"  Global scene graphs: {global_file}")
    print(f"  Query scene graphs: {query_file}")
    print(f"  Max samples: {max_samples}")
    print(f"  Matching strategy: original_dataset_index")
    print(f"  Choices source: Scene graph JSON files")
    
    if not os.path.exists(global_file):
        print(f"ERROR: {global_file} not found!")
        return
    
    if not os.path.exists(query_file):
        print(f"ERROR: {query_file} not found!")
        return
    
    try:
        evaluator = MMBenchVQAEvaluator(
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
        print("The matching key should be: original_dataset_index")
        print("Check that your scene graph files have original_dataset_index fields")
        return
    
    if matching_samples:
        print(f"\nChoice loading check (first sample):")
        first_sample = matching_samples[0]
        has_choices = any([
            first_sample.get('choice_a', '').strip(),
            first_sample.get('choice_b', '').strip(),
            first_sample.get('choice_c', '').strip(),
            first_sample.get('choice_d', '').strip()
        ])
        print(f"  Sample {first_sample['original_dataset_index']} has choices: {has_choices}")
        if has_choices:
            print(f"    A: {first_sample.get('choice_a', '')[:50]}...")
            print(f"    B: {first_sample.get('choice_b', '')[:50]}...")
            print(f"    C: {first_sample.get('choice_c', '')[:50]}...")
            print(f"    D: {first_sample.get('choice_d', '')[:50]}...")
        else:
            print("  Warning: No choices found in scene graph files!")
            print("  This may indicate the JSON structure doesn't contain choice fields.")
    
    try:
        results = evaluator.evaluate_dataset(max_samples=max_samples)
        
        if results:
            print(f"\nEvaluation Summary:")
            print(f"  Final Accuracy: {results['accuracy']:.4f}")
            print(f"  Results Directory: {results['results_dir']}")
            
            if results['results']:
                print(f"\nSample Results:")
                for i, result in enumerate(results['results'][:3]):
                    status = "CORRECT" if result['is_correct'] else "INCORRECT"
                    print(f"  {status} Sample {i+1} (Index: {result['original_dataset_index']}):")
                    print(f"    Question: {result['question'][:60]}...")
                    print(f"    Category: {result['category']}")
                    print(f"    Choices: A={result['choice_a'][:20]}..., B={result['choice_b'][:20]}...")
                    print(f"    Ground Truth: {result['ground_truth']}")
                    print(f"    Predicted: {result['predicted_answer']}")
            
            if 'category_accuracies' in results:
                cat_acc = results['category_accuracies']
                if cat_acc:
                    sorted_categories = sorted(cat_acc.items(), key=lambda x: x[1]['accuracy'], reverse=True)
                    print(f"\nTop Performing Categories:")
                    for cat, acc_data in sorted_categories[:3]:
                        print(f"  {cat}: {acc_data['accuracy']:.4f} ({acc_data['correct']}/{acc_data['total']})")
                    
                    print(f"\nBottom Performing Categories:")
                    for cat, acc_data in sorted_categories[-3:]:
                        print(f"  {cat}: {acc_data['accuracy']:.4f} ({acc_data['correct']}/{acc_data['total']})")
        
    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()