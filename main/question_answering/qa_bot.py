from transformers import AutoTokenizer, AutoModelForCausalLM

with open(r'src\question_answering\prompt.txt', 'r', encoding='utf-8') as f:
    prompt = f.read()

class QABot:
    def __init__(self, modelqa_path):
        self.model_path = modelqa_path
        self.prompt = prompt

    def load_model(self):
        """Tải mô hình và tokenizer từ đường dẫn."""
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def generate_answer(self,content,query):
        """Tạo câu trả lời dựa trên truy vấn của người dùng."""
        input_text = f"{self.prompt}\n\nUser: {query}\n\nContext: {content}\n\nAnswer"
        
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")

        outputs = self.model.generate(
            inputs,
            max_length=100000,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
