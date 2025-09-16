import torch
import os
from tqdm import tqdm
import json
from datetime import datetime
from PIL import Image
import re
import gc
from transformers import AutoProcessor, LlavaForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import ast

from config import Config
from data_loader import v_get_dataloader_hf

class WHOOPSVQAEvaluator:
    """VQA evaluator for WHOOPS dataset using scene graphs"""
    
    def __init__(self, 
                 global_scene_graphs_path="cache/scene_graphs/test_global_scene_graphs_explained.json",
                 query_scene_graphs_path="cache/scene_graphs/test_scene_graphs (3)_explained_query_specific.json"):
        
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
        
        print("Loading WHOOPS dataset...")
        self.dataloader = None
        self.image_lookup = {}
        self._load_whoops_dataset()
        
        print("Loading BERT model for answer evaluation...")
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("BERT model loaded successfully")
    
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
            
            if "ASSISTANT:" in cleaned:
                after_assistant = cleaned.split("ASSISTANT:")[-1].strip()
                extracted = self._extract_json_from_text(after_assistant)
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
    
    def _load_global_scene_graphs(self, path):
        """Load global scene graphs"""
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                global_data = json.load(f)
            
            print(f"Processing {len(global_data)} global scene graph items...")
            success_count = 0
            
            for i, item in enumerate(global_data):
                try:
                    image_id = str(item.get('image_id', ''))
                    
                    if not image_id:
                        continue
                    
                    global_sg_raw = item.get('global_scene_graph', '')
                    scene_graph = self._extract_and_validate_scene_graph(global_sg_raw)
                    
                    if not scene_graph:
                        scene_graph = {"objects": [{"name": "Unknown", "attributes": []}], "relationships": []}
                    
                    self.global_lookup[image_id] = {
                        'image_id': image_id,
                        'global_scene_graph': json.dumps(scene_graph),
                        'commonsense_category': item.get('commonsense_category', ''),
                        'designer_explanation': item.get('designer_explanation', ''),
                        'selected_caption': item.get('selected_caption', ''),
                        'qa_pairs': item.get('qa_pairs', []),
                        'raw_data': item
                    }
                    success_count += 1
                    
                except Exception as e:
                    print(f"Error processing global item {i}: {e}")
                    continue
            
            print(f"Loaded global scene graphs for {success_count} samples")
            
        except Exception as e:
            print(f"Error loading global scene graphs: {e}")
    
    def _load_query_scene_graphs(self, path):
        """Load query-specific scene graphs"""
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
                    image_id = str(item.get('image_id', ''))
                    question = item.get('question', '')
                    
                    if not image_id or not question:
                        continue
                    
                    query_sg_raw = item.get('scene_graph', '')
                    scene_graph = self._extract_and_validate_scene_graph(query_sg_raw)
                    
                    if not scene_graph:
                        scene_graph = {"objects": [{"name": "Unknown", "attributes": []}], "relationships": []}
                    
                    if image_id not in self.query_lookup:
                        self.query_lookup[image_id] = {}
                    
                    self.query_lookup[image_id][question] = {
                        'query_scene_graph': json.dumps(scene_graph),
                        'ground_truth': item.get('ground_truth', ''),
                        'qa_pair_index': item.get('qa_pair_index', 0),
                        'scene_graph_explanation': item.get('scene_graph_explanation', ''),
                        'raw_data': item
                    }
                    success_count += 1
                    
                except Exception as e:
                    print(f"Error processing query item {i}: {e}")
                    continue
            
            print(f"Loaded query-specific scene graphs for {success_count} samples")
            
        except Exception as e:
            print(f"Error loading query scene graphs: {e}")
    
    def _load_whoops_dataset(self):
        """Load WHOOPS dataset and build lookups"""
        try:
            print("Loading WHOOPS dataset...")
            
            Image.MAX_IMAGE_PIXELS = None
            
            self.dataloader = v_get_dataloader_hf(
                split="test",
                batch_size=1,
                shuffle=False
            )
            
            print("Building image lookup from dataloader...")
            
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Building image lookup")):
                try:
                    if 'image' in batch:
                        image = batch['image']
                        if isinstance(image, list) and len(image) > 0:
                            image = image[0]
                    else:
                        continue
                    
                    image_id = None
                    for key in ['image_id', 'id', 'image_name', 'filename']:
                        if key in batch and batch[key] is not None:
                            if isinstance(batch[key], list):
                                image_id = str(batch[key][0])
                            else:
                                image_id = str(batch[key])
                            break
                    
                    if image_id is None:
                        image_id = str(batch_idx)
                    
                    processed_image = self.preprocess_image(image)
                    
                    self.image_lookup[image_id] = {
                        'image': processed_image,
                        'image_id': image_id,
                        'qa_pairs': batch.get('qa_pairs', [[]])[0] if batch.get('qa_pairs') else [],
                        'commonsense_category': batch.get('commonsense_category', [''])[0] if batch.get('commonsense_category') else '',
                        'designer_explanation': batch.get('designer_explanation', [''])[0] if batch.get('designer_explanation') else '',
                        'selected_caption': batch.get('selected_caption', [''])[0] if batch.get('selected_caption') else ''
                    }
                    
                    clean_id = image_id.replace('.jpg', '').replace('.png', '').strip()
                    self.image_lookup[clean_id] = self.image_lookup[image_id]
                    self.image_lookup[str(batch_idx)] = self.image_lookup[image_id]
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue
            
            print(f"Created image lookup for {len(self.image_lookup)} images")
            
        except Exception as e:
            print(f"Error loading WHOOPS dataset: {e}")
            self.dataloader = None
            self.image_lookup = {}
    
    def preprocess_image(self, image):
        """Ensure image is in correct format for VLM model"""
        try:
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image.squeeze(0)
                if image.dim() == 3 and image.shape[0] in [1, 3]:
                    if image.min() < 0:
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        image = image * std + mean
                    
                    image = torch.clamp(image, 0, 1)
                    image_np = image.permute(1, 2, 0).cpu().numpy()
                    image_np = (image_np * 255).astype(np.uint8)
                    image = Image.fromarray(image_np)
            
            elif isinstance(image, np.ndarray):
                if image.dtype == np.float32 or image.dtype == np.float64:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                if len(image.shape) == 3 and image.shape[0] in [1, 3]:
                    image = np.transpose(image, (1, 2, 0))
                
                image = Image.fromarray(image)
            
            if not isinstance(image, Image.Image):
                raise ValueError(f"Could not convert image to PIL format. Got type: {type(image)}")
            
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
            elif image.mode == 'L':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return Image.new('RGB', (224, 224), color='black')
    
    def get_sample_data(self, image_id):
        """Get sample data including image from WHOOPS dataset"""
        image_id_str = str(image_id)
        
        if image_id_str in self.image_lookup:
            return self.image_lookup[image_id_str]
        
        clean_id = image_id_str.replace('.jpg', '').replace('.png', '').strip()
        if clean_id in self.image_lookup:
            return self.image_lookup[clean_id]
        
        for stored_id in self.image_lookup.keys():
            clean_stored_id = stored_id.replace('.jpg', '').replace('.png', '').strip()
            if clean_stored_id == clean_id:
                return self.image_lookup[stored_id]
        
        return None
    
    def create_weighted_vqa_prompt(self, global_sg, query_sg, question):
        """Create VQA prompt using both scene graphs (adapted from zero_shot.py)"""
        
        context = "This image contains something unusual or unexpected. "
        extraction = "Answer with ONLY one or two words. Do not provide explanations or full sentences."
        
        prompt = f"""USER: <image>

Global Scene Graph:
{global_sg}
Query Specific Scene Graph:
{query_sg}

Question: {question}

INSTRUCTION: Use both the query and global scene graphs to answer the question about the image.
{extraction}
ASSISTANT:"""
        
        return prompt
    
    def generate_vqa_response(self, image, global_sg, query_sg, question):
        """Generate VQA response using both scene graphs with actual image"""
        try:
            prompt = self.create_weighted_vqa_prompt(global_sg, query_sg, question)
            
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
            
            response_words = response.split()[:2]
            response = ' '.join(response_words)
            
            return response
            
        except Exception as e:
            print(f"Error generating VQA response: {e}")
            return f"Error: {str(e)}"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def calculate_bert_similarity(self, predicted, ground_truth):
        """Calculate BERT-based similarity score between predicted and ground truth answers"""
        try:
            if not predicted or not ground_truth or predicted.startswith("Error"):
                return 0.0
            
            predicted_clean = str(predicted).strip().lower()
            ground_truth_clean = str(ground_truth).strip().lower()
            
            if predicted_clean == ground_truth_clean:
                return 1.0
            
            embeddings = self.bert_model.encode([predicted_clean, ground_truth_clean])
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            similarity = max(0.0, min(1.0, similarity))
            
            return similarity
            
        except Exception as e:
            print(f"Error calculating BERT similarity: {e}")
            return 0.0
    
    def get_scene_graphs_for_question(self, image_id, question):
        """Get both query-specific and global scene graphs for a specific image and question"""
        image_id_str = str(image_id)
        
        query_sg = '{"objects": [], "relationships": [], "error": "No query-specific scene graph available"}'
        query_explanation = ""
        
        if image_id_str in self.query_lookup:
            image_queries = self.query_lookup[image_id_str]
            
            if question in image_queries:
                query_sg = image_queries[question]['query_scene_graph']
                query_explanation = image_queries[question].get('scene_graph_explanation', '')
            else:
                question_lower = question.lower().strip()
                for stored_question, data in image_queries.items():
                    if stored_question.lower().strip() == question_lower:
                        query_sg = data['query_scene_graph']
                        query_explanation = data.get('scene_graph_explanation', '')
                        break
        
        global_sg = '{"objects": [], "relationships": [], "error": "No global scene graph available"}'
        global_explanation = ""
        
        if image_id_str in self.global_lookup:
            global_data = self.global_lookup[image_id_str]
            global_sg = global_data.get('global_scene_graph', global_sg)
            global_explanation = global_data.get('global_scene_graph_explanation', '')
        
        return query_sg, query_explanation, global_sg, global_explanation
    
    def get_matching_samples(self):
        """Get samples that have both scene graphs and exist in dataset"""
        print(f"Finding matching samples...")
        print(f"  Global lookup: {len(self.global_lookup)} items")
        print(f"  Query lookup: {len(self.query_lookup)} items")  
        print(f"  Dataset lookup: {len(self.image_lookup)} items")
        
        matching_samples = []
        
        global_keys = set(self.global_lookup.keys())
        query_keys = set(self.query_lookup.keys())
        dataset_keys = set(self.image_lookup.keys())
        
        final_common = global_keys & query_keys & dataset_keys
        
        print(f"  Final matching (all three sources): {len(final_common)}")
        
        for image_id in final_common:
            image_data = self.image_lookup[image_id]
            
            if image_id in self.query_lookup:
                for question, query_data in self.query_lookup[image_id].items():
                    matching_samples.append({
                        'image_id': image_id,
                        'question': question,
                        'ground_truth': query_data['ground_truth'],
                        'qa_pair_index': query_data['qa_pair_index'],
                        'commonsense_category': image_data.get('commonsense_category', ''),
                        'designer_explanation': image_data.get('designer_explanation', ''),
                        'selected_caption': image_data.get('selected_caption', '')
                    })
        
        print(f"Final matching samples: {len(matching_samples)}")
        
        return matching_samples
    
    def evaluate_dataset(self, max_samples=None):
        """Evaluate VQA using both scene graphs on WHOOPS dataset"""
        print("Starting WHOOPS VQA Evaluation with Scene Graphs")
        print("="*60)
        
        results = []
        total_bert_score = 0.0
        total_qa_pairs = 0
        processed_samples = 0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"whoops_vqa_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        matching_samples = self.get_matching_samples()
        
        if len(matching_samples) == 0:
            print("ERROR: No matching samples found!")
            return None
        
        samples_to_process = matching_samples[:max_samples] if max_samples else matching_samples
        print(f"Processing {len(samples_to_process)} samples...")
        
        category_stats = {}
        
        for i, sample_data in enumerate(tqdm(samples_to_process, desc="VQA Evaluation")):
            image_id = sample_data['image_id']
            question = sample_data['question']
            ground_truth = sample_data['ground_truth']
            category = sample_data.get('commonsense_category', 'unknown')
            
            dataset_sample = self.get_sample_data(image_id)
            if not dataset_sample or not dataset_sample['image']:
                print(f"Warning: Could not get image for {image_id}, skipping...")
                continue
            
            try:
                image = dataset_sample['image']
                
                query_sg, query_explanation, global_sg, global_explanation = self.get_scene_graphs_for_question(image_id, question)
                
                predicted_answer = self.generate_vqa_response(image, global_sg, query_sg, question)
                
                bert_score = self.calculate_bert_similarity(predicted_answer, ground_truth)
                
                if category not in category_stats:
                    category_stats[category] = {'bert_scores': [], 'count': 0}
                category_stats[category]['bert_scores'].append(bert_score)
                category_stats[category]['count'] += 1
                
                result = {
                    'image_id': image_id,
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted_answer,
                    'bert_score': bert_score,
                    'category': category,
                    'qa_pair_index': sample_data['qa_pair_index'],
                    'designer_explanation': sample_data.get('designer_explanation', ''),
                    'selected_caption': sample_data.get('selected_caption', ''),
                    'global_scene_graph_preview': global_sg[:200] + "..." if len(global_sg) > 200 else global_sg,
                    'query_scene_graph_preview': query_sg[:200] + "..." if len(query_sg) > 200 else query_sg
                }
                
                results.append(result)
                
                total_bert_score += bert_score
                total_qa_pairs += 1
                processed_samples += 1
                
                if processed_samples % 10 == 0:
                    current_avg_bert = total_bert_score / total_qa_pairs
                    print(f"   Processed {processed_samples}/{len(samples_to_process)} samples, Avg BERT Score: {current_avg_bert:.4f}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                if processed_samples % 50 == 0:
                    intermediate_file = os.path.join(results_dir, f'intermediate_results_{processed_samples}.json')
                    with open(intermediate_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
            
            except Exception as e:
                print(f"Error processing sample {image_id}: {e}")
                continue
        
        final_avg_bert = total_bert_score / total_qa_pairs if total_qa_pairs > 0 else 0
        
        category_averages = {}
        for cat, stats in category_stats.items():
            if stats['count'] > 0:
                category_averages[cat] = {
                    'average_bert_score': sum(stats['bert_scores']) / len(stats['bert_scores']),
                    'count': stats['count']
                }
        
        detailed_file = os.path.join(results_dir, 'detailed_results.json')
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        summary = {
            'timestamp': timestamp,
            'dataset': 'WHOOPS',
            'evaluation_type': 'VQA with Scene Graphs',
            'total_samples': processed_samples,
            'total_qa_pairs': total_qa_pairs,
            'average_bert_score': final_avg_bert,
            'category_averages': category_averages,
            'methodology': 'VQA using global and query-specific scene graphs with images',
            'evaluation_metric': 'BERT Similarity (BEM)',
            'config': {
                'max_samples': max_samples,
                'model': 'llava-hf/llava-1.5-7b-hf',
                'bert_model': 'all-MiniLM-L6-v2',
                'global_scene_graphs_loaded': len(self.global_lookup),
                'query_scene_graphs_loaded': len(self.query_lookup),
                'dataset_samples_loaded': len(self.image_lookup),
                'matching_samples_found': len(matching_samples)
            }
        }
        
        summary_file = os.path.join(results_dir, 'evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nWHOOPS VQA evaluation completed!")
        print(f"Results Summary:")
        print(f"   Total Samples: {processed_samples}")
        print(f"   Total Q&A Pairs: {total_qa_pairs}")
        print(f"   Average BERT Score: {final_avg_bert:.4f}")
        print(f"   Results saved to: {results_dir}")
        
        print(f"\nCategory-wise Performance:")
        for cat, data in sorted(category_averages.items(), key=lambda x: x[1]['average_bert_score'], reverse=True):
            print(f"   {cat}: {data['average_bert_score']:.4f} ({data['count']} samples)")
        
        return {
            'bert_score': final_avg_bert,
            'results': results,
            'summary': summary,
            'results_dir': results_dir,
            'category_averages': category_averages
        }

def main():
    """Main execution function"""
    print("WHOOPS VQA Evaluation with Scene Graphs")
    print("="*50)
    
    global_file = "cache/scene_graphs/test_global_scene_graphs_explained.json"
    query_file = "cache/scene_graphs/test_scene_graphs (3)_explained_query_specific.json"
    max_samples = 1000
    
    print(f"Configuration:")
    print(f"  Global scene graphs: {global_file}")
    print(f"  Query scene graphs: {query_file}")
    print(f"  Max samples: {max_samples}")
    print(f"  Dataset: WHOOPS")
    
    if not os.path.exists(global_file):
        print(f"ERROR: {global_file} not found!")
        return
    
    if not os.path.exists(query_file):
        print(f"ERROR: {query_file} not found!")
        return
    
    try:
        evaluator = WHOOPSVQAEvaluator(
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
    print(f"  Dataset samples: {len(evaluator.image_lookup)}")
    
    if len(evaluator.global_lookup) == 0:
        print("ERROR: No global scene graphs loaded!")
        return
    
    if len(evaluator.query_lookup) == 0:
        print("ERROR: No query scene graphs loaded!")
        return
    
    if len(evaluator.image_lookup) == 0:
        print("ERROR: No dataset samples loaded!")
        return
    
    matching_samples = evaluator.get_matching_samples()
    
    if len(matching_samples) == 0:
        print("ERROR: No matching samples found!")
        return
    
    try:
        results = evaluator.evaluate_dataset(max_samples=max_samples)
        
        if results:
            print(f"\nEvaluation Summary:")
            print(f"  Average BERT Score: {results['bert_score']:.4f}")
            print(f"  Results Directory: {results['results_dir']}")
            
            if results['results']:
                print(f"\nSample Results:")
                for i, result in enumerate(results['results'][:3]):
                    print(f"  Sample {i+1} (ID: {result['image_id']}):")
                    print(f"    Question: {result['question']}")
                    print(f"    Category: {result['category']}")
                    print(f"    Ground Truth: {result['ground_truth']}")
                    print(f"    Predicted: {result['predicted_answer']}")
                    print(f"    BERT Score: {result['bert_score']:.4f}")
            
            if 'category_averages' in results:
                cat_avg = results['category_averages']
                if cat_avg:
                    sorted_categories = sorted(cat_avg.items(), key=lambda x: x[1]['average_bert_score'], reverse=True)
                    print(f"\nTop Performing Categories:")
                    for cat, data in sorted_categories[:3]:
                        print(f"  {cat}: {data['average_bert_score']:.4f} ({data['count']} samples)")
                    
                    print(f"\nBottom Performing Categories:")
                    for cat, data in sorted_categories[-3:]:
                        print(f"  {cat}: {data['average_bert_score']:.4f} ({data['count']} samples)")
        
    except Exception as e:
        print(f"ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()