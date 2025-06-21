import numpy as np
from src.core.advanced_trainer import AdvancedTrainer
from src.core.letter_image_generator import LetterImageGenerator
import string

INPUT_SHAPE = (28, 28)
SDR_DIM = 256
LATENT_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 2
HIDDEN_DIM = 64
EPOCHS = 1000
LETTERS = list(string.digits)

generator = LetterImageGenerator(*INPUT_SHAPE)
trainer = AdvancedTrainer(INPUT_SHAPE, SDR_DIM, LATENT_DIM, NUM_LAYERS, NUM_HEADS, HIDDEN_DIM)

for letter in LETTERS:
    print(f"\n[Training] Starting for letter: '{letter}'")
    for epoch in range(EPOCHS):
        img = generator.get_image(letter, augment=True)
        result = trainer.train_step(img)
        error = np.abs(result['error']).mean()
        print(f"Epoch {epoch+1} | Letter: {letter} | Error: {error:.4f}")
