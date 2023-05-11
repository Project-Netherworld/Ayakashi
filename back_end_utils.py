import base64
import json
import torch
def decode_encoded_tokenized_tensor(a_encoded_tokens):
    """
    Decode a base64 encoded tokenized tensor into a tensor.

    :description: In this function's case, it decodes a base64 encoded tokenized tensor into a tensor, such that
    it can be fed into the backend's model. It does not detokenize the list, as there is no tokenizer on the backend.
    :param a_encoded_tokens: The encoded base64 string sent back from the backend to decode.
    :return: The tokenized chat history.
    :rtype: torch.Tensor (varies on which specific type)
    :acknowledgements: Antony Mercurio, who recommended me to utilize this method and gave me portions of the code.
    """
    partial_decoded_history = base64.b64decode(a_encoded_tokens)
    list_chat_history = json.loads(partial_decoded_history.decode("utf-8"))
    return torch.tensor([list_chat_history])

def get_encoded_str_from_token_list(a_message):
    """
    Gets the base 64 encoded string given a list of tokens.

    :description: Encodes a list of tokens into a base 64 string. Why? Simple. Easy serialization of complex data.
    My friend, Anthony Mercurio, recommended me this as he saw it being used by a start-up named NovelAI for
    performance reasons. I can see why, as the only other way of serializing a stupidly large list would be using
    the python pickle library or making a large multidimensional list, which would require quite the lot of string
    manipulation. So, I chose the lesser of the two evils and decided to use base64 as my serialization algorithm.

    :algorithm: 1. First, the list itself is converted to a json readable string, so it can be broken down into raw bytes.
    2. The string is then turned into bytes to prepare it for base64 encoding.
    3. Then, the encoded base64 are turned into a string.
    4. This might seem counterintuitive, but the string is then decoded for utf-8. Why? So it can be sent via json.
       Otherwise, there'd be no way to send it over, and I've confirmed the string itself is still base64 encoded.
    :param a_message: The list of tokens representing an arbitrary message. Typically, is the entire chat history in
    practice, however.
    :return: The original message now encoded into base64.
    :rtype: str
    :acknowledgements: Antony Mercurio, who recommended me to utilize this method and gave me portions of the code.
    """
    encoded_str = json.dumps(
        base64.b64encode(bytes(json.dumps(a_message.tolist()), encoding='utf-8')).decode("utf-8"))
    return encoded_str
