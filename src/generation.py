import random

class TextGenerator:
    def __init__(self, model):
        self.model = model

    def greedy_decoding(self, input_text, max_length):
        output = input_text
        for _ in range(max_length):
            predicted_token = self.model.predict_next_token(output)
            output += predicted_token
            if predicted_token == '<EOS>':  # End of sequence token
                break
        return output

    def temperature_sampling(self, input_text, max_length, temperature=1.0):
        output = input_text
        for _ in range(max_length):
            predicted_tokens = self.model.predict_next_tokens(output)
            token_probs = self.model.get_token_probabilities(predicted_tokens)
            token_probs = [prob ** (1.0 / temperature) for prob in token_probs]  # apply temperature
            token_probs_sum = sum(token_probs)
            token_probs = [prob / token_probs_sum for prob in token_probs]  # normalize
            next_token = random.choices(predicted_tokens, weights=token_probs)[0]
            output += next_token
            if next_token == '<EOS>':
                break
        return output

# Example usage, assuming a model instance is available:
# generator = TextGenerator(model)
# print(generator.greedy_decoding("Once upon a time", 50))
# print(generator.temperature_sampling("Once upon a time", 50, temperature=0.7))