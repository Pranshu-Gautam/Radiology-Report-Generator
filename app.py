from flask import Flask, request, jsonify, send_from_directory,render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'radiology_report_generator_epoch3.pt'  # Replace with your model checkpoint path

# Define the image transformation pipeline
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define the model class
class RadiologyReportGenerator(torch.nn.Module):
    def __init__(self, encoder_model_name='resnet50', decoder_model_name='gpt2'):
        super(RadiologyReportGenerator, self).__init__()
        # Encoder: Pretrained ResNet
        self.encoder = torch.hub.load('pytorch/vision:v0.10.0', encoder_model_name, pretrained=True)
        self.encoder = torch.nn.Sequential(*list(self.encoder.children())[:-1])  # Remove classification layer
        self.encoder_dim = 2048
        self.decoder_embedding_dim = 768  # GPT-2's embedding size
        self.fc = torch.nn.Linear(self.encoder_dim, self.decoder_embedding_dim)

        # Decoder: Pretrained GPT-2
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_model_name)
        self.decoder.resize_token_embeddings(len(tokenizer))  # Resize tokenizer embeddings

    def forward(self, images, input_ids, attention_mask, labels=None):
        # Process image through encoder
        features = self.encoder(images).view(images.size(0), -1)
        features = self.fc(features)

        # Append image features as prefix to decoder input embeddings
        decoder_inputs_embeds = self.decoder.transformer.wte(input_ids)
        image_features = features.unsqueeze(1)
        decoder_inputs_embeds = torch.cat((image_features, decoder_inputs_embeds), dim=1)
        image_attention = torch.ones((attention_mask.size(0), 1), device=device)
        attention_mask = torch.cat((image_attention, attention_mask), dim=1)

        if labels is not None:
            labels = torch.cat((torch.full((labels.size(0), 1), -100, dtype=torch.long, device=device), labels), dim=1)

        outputs = self.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load the model
model = RadiologyReportGenerator().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def generate_report(image, max_length=100):
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)
        features = model.encoder(image).view(1, -1)
        features = model.fc(features)

        generated_ids = model.decoder.generate(
            inputs_embeds=features.unsqueeze(1),
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    try:
        # Load and preprocess image
        image = Image.open(file).convert('RGB')
        transformed_image = image_transform(image)

        # Generate report
        report = generate_report(transformed_image)
        return jsonify({'report': report})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
