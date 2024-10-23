from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Cargar el modelo y el tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

@app.route('/generar', methods=['POST'])
def generar_texto():
    data = request.json
    prompt = data.get('prompt', '')
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50)
    texto_generado = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'respuesta': texto_generado})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
