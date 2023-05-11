from typing import List, Tuple

import torch
from transformers import LogitsProcessor, LogitsWarper
from math import exp


# Note that despite borrowing from many sources, these are documented as if they were my functions. Why?
# Because otherwise this would look like pig latin to the naked eye. Even if the algorithms are not to 1-1 with the
# implementation, you can get a rough idea using the documentation.

class TailFreeSamplingLogitsWarper(LogitsWarper):
    """
    Performs tail free sampling, as described in
    https://www.trentonbricken.com/Tail-Free-Sampling/.
    """
    def __init__(self, tfs: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        """
        Ctor for TailFreeSamplingLogitsWarper class. Defines the __init__ abstract function from the
        inherited class LogitsWarper.

        :param tfs: If set to < 1, only the most probable tokens where the second derivative of the probabilities of
        the tokens sorted in descending order of probability add up to at most :obj:`tfs` are kept for generation.
        :type tfs: float
        :param filter_value: All filtered values will be set to this float value.
        :type filter_value: float
        :param min_tokens_to_keep: Minimum number of tokens that cannot be filtered.
        :type min_tokens_to_keep: int
        :acknowledgements This function was adapted from
        https://github.com/hitomi-team/sukima/blob/main/app/gpt/warpers.py,
        which implements Tail Free Sampling, as described here:  https://www.trentonbricken.com/Tail-Free-Sampling/.
        """
        tfs = float(tfs)
        if tfs < 0 or tfs > 1.0:
            raise ValueError(f"`tfs` has to be a float > 0 and < 1, but is {tfs}")

        self.tfs = tfs
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Tail Free samples input_ids. Defines the __call__ abstract function from the inherited class LogitsWarper.

        :description: This defines the function LogitsWarper.__call__ from the transformers library. During the process
        of the calling, sampling is performed on the parameter input_ids, which is the tokenized chat history in this
        case and/or response. As the algorithm itself is complicated, it'll be paraphrased from a source listed
        under the acknowledgements tab(Kobold AI & Trenton Bricken). Here it is below:
        :algorithm: 1. Apply the softmax function to the logits to get their corresponding probabilities.
        2. Sort the tokens in descending order of probability (i.e. token with highest probability comes first and token
        with the lowest probability comes last).
        3. Calculate the first differences of these sorted probabilities.
        4. Then calculate the second differences (the differences of the first differences).
        5. Take the absolute values of the second differences.
        6. Divide these values by their sum.
        7. Compute the cumulative sums of these values.
        8. Add a 0 at the beginning and a 1 at the end.
        9. Remove the tokens whose value as computed in the previous step is greater than the tail free sampling value
        (by setting their logits to negative infinity).
        In practice, this sampler makes responses more creative at the expense of coherency.
        :param input_ids: The inputted token ids. In this case, the tokenized chat history.
        :type input_ids: torch.LongTensor
        :param scores: The modified version of input_ids after sampling is performed.
        :type scores: torch.FloatTensor
        :return: The modified version of input_ids after sampling is performed.
        :rtype: torch.FloatTensor
        :acknowledgements This function was adapted from
        https://github.com/hitomi-team/sukima/blob/main/app/gpt/warpers.py,
        which implements Tail Free Sampling, as described here: https://www.trentonbricken.com/Tail-Free-Sampling/. In
        addition, the algorithm summary above comes from Trenton Bricken's tensorflow implementation, which is then
        further paraphrased in another text generation/chatbot client named Kobold AI here:
        https://github.com/BlinkDL/RWKV-LM/tree/4cb363e5aa31978d801a47bc89d28e927ab6912e#tail-free-sampling
        """
        if self.filter_value >= 1.0:
            return scores

        # 2) sort the tokens in descending order of probability
        # Note that this is technically out of order compared to the description, however, I will assume the order
        # does not matter. In the odd case that it does, I will not touch this code.
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        # 1) apply softmax function to get logit probabilites
        probs = sorted_logits.softmax(dim=-1)

        # 3,4, & 5) Compute second derivative normalized CDF (take first and second deriative and abs val it)
        d2 = probs.diff().diff().abs()
        # 6) Divide these values by their sum.
        normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
        # 7) Compute the cumulative sum of these values.
        normalized_d2_cdf = normalized_d2.cumsum(dim=-1)

        # 8 & 9) Remove tokens with CDF value above the threshold (token with 0 are kept)
        # Again, the implementation differs slightly from the algorithm, so keep that in mind.
        sorted_indices_to_remove = normalized_d2_cdf > self.tfs
        # Centre the distribution around the cutoff as in the original implementation of the algorithm
        sorted_indices_to_remove = torch.cat(
            (
                torch.zeros(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
                sorted_indices_to_remove,
                torch.ones(scores.shape[0], 1, dtype=torch.bool, device=scores.device),
            ),
            dim=-1,
        )
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

class TopALogitsWarper(LogitsWarper):
    """
    Class that implents top_a sampling as described in:
    https://github.com/BlinkDL/RWKV-LM/tree/4cb363e5aa31978d801a47bc89d28e927ab6912e#the-top-a-sampling-method
    """
    def __init__(self, threshold: float, filter_value: float = -float("inf")):
        """
        Ctor for the TopALogitsWarper Class, which inherits from the LogitsWarper class.

        :param threshold: The threshold of tokens that must add up to be kept. This is alternatively known as the value
        of a in top_a sampling.
        Note that at the highest probability token must be kept at all times in spite of this.
        :type threshold: float
        :param filter_value: The value of which the tokens below the threshold are to be given the value of.
        :type filter_value: float
        :acknowledgements This function was adapted from
        https://github.com/hitomi-team/sukima/blob/main/app/gpt/warpers.py.
        The description of the function is paraphrased from this link, from a similar text generation/chatbot project:
        https://github.com/KoboldAI/KoboldAI-Client/wiki/Settings#top-a-sampling.
        The idea for the algorithm is defined here:
        https://github.com/BlinkDL/RWKV-LM/tree/4cb363e5aa31978d801a47bc89d28e927ab6912e#the-top-a-sampling-method
        """
        if not isinstance(threshold, float) or (threshold < 0 or threshold > 1.0):
            raise ValueError(f"`threshold` has to be a float > 0 and < 1, but is {threshold}")

        self.z = threshold
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Performs top_a sampling on the input tensor. Implements the __call__ function from LogitsWarper.

        :description: This function performs top_a sampling on the input ids, or otherwise the tokenized chat history.
        The sampling in question utilizes a math function ax^2, in where a is self.threshold, and x is the probability
        that the next token should show up given the inputted chat history. Anything lower than ax^2 will be removed,
        unless all the tokens are removed, in which case, the most probable token is kept. The algorithm (taken from
        another source that will be mentioned) is as follows:
        1. Find the max probability probs after softmax.
        2. Remove all entries whose probability is lower than 0.2 * pow(probs, 2). So it's adaptive, hence "top-a".
        3. (Optional)Feel free to tune the 0.2 and 2 factor. Tune 0.2 first.
        For example,
        If self.threshold=0.9, then remove all tokens with prob < 0.162 (so, removing all alternatives)
        If self.threshold=0.5, then remove all tokens with prob < 0.05 (so, allowing more choices)
        If self.threshold=0.1, then remove all tokens with prob < 0.002 (so, allowing lots of possibilities).
        This sampler is useful as it can be a slightly different and more powerful version of top_p, which samples in a
        similar way (getting rid of tokens that are below a certain threshold).
        :param input_ids: The inputted token ids. In this case, the tokenized chat history.
        :type torch.LongTensor
        :param scores:The modified version of input_ids after sampling is performed.
        :return: a torch.LongTensor, representing the modified version of input_ids after sampling is performed.
        :acknowledgements: This function was adapted from
        https://github.com/hitomi-team/sukima/blob/main/app/gpt/warpers.py.
        The description of the function is paraphrased from this link, from a similar text generation/chatbot project:
        https://github.com/KoboldAI/KoboldAI-Client/wiki/Settings#top-a-sampling.
        The idea for the algorithm is defined here, alongside the description of the alogorithm:
        https://github.com/BlinkDL/RWKV-LM/tree/4cb363e5aa31978d801a47bc89d28e927ab6912e#the-top-a-sampling-method
        """

        #Gather probabilites from softmax.
        probs = torch.nn.functional.softmax(scores, dim=-1)
        # 1) Find the max probability limit after softmax.
        limit = torch.pow(torch.max(probs), 2.0) * self.z
        # 2) Remove all entries whose probability is lower than 0.2*pow(probs).
        indices_to_remove = probs < limit
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class LogitBiasProcessor(LogitsProcessor):
    """
    Class that implements Logit Biasing as described in:
    https://help.openai.com/en/articles/5247780-using-logit-bias-to-define-token-probability
    """
    def __init__(self, logit_bias: List[Tuple[int, float]] = []):
        """
        :name - __init__ the ctor for the LogitBiasProcessor class that inherits from the LogitsProcessor class.
        :param logit_bias: The list of logits to bias, alongside their biases.
        "Adds a float bias to the given token's logit" - sukima.
        :type List[Tuple[int, float]]
        :acknowledgements - This function was adapted from
        https://github.com/hitomi-team/sukima/blob/main/app/gpt/warpers.py.
        """
        if not isinstance(logit_bias, list) and len(logit_bias) > 0:
            raise ValueError("`logit_bias` has to be a non-empty list")

        # apply exp to each bias
        self.logit_bias = [(token, exp(bias)) for token, bias in logit_bias]
        self.bias = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Processes the input_ids such that it biases certain tokens. Implements the __call__ function
        from the class LogitsProcessor.

        :description - This following command adds a bias to certain tokens given a list of input ids, or, otherwise,
        the tokenized chat history. While I can't seem to find a documentation for this processor, the algorithm
        should give readers a general idea. This processor is particularly useful for targeting puncutation.
        Let's say a user wants to make a character  use "!" a lot.
        By passing the tokenized version of "!" and a threshold, this function can help "!" appear more frequently
        by biasing it.

        :algorithm: 1. Create a new tensor that is the same size as the columns of the scores tensor, representing the modified
           chat history.
        2. Fill in the tensor with self.logit_biases to ensure the tokens that are biased get their extra bias value.
        3. Standardize the tensor such that it's the same dimensions as the input_ids, and convert to the backend's
        selected device. 
        4. The following always executes as the last step of the function call:
        Add the scores tensor and the bias tensor.

        :param input_ids: The inputted token ids. In this case, the tokenized chat history.
        :type input_ids: torch.LongTensor
        :param scores:The modified version of input_ids after sampling is performed.
        :type scores: torch.FloatTensor
        :return: The modified version of input_ids after sampling is performed.
        :rtype: torch.LongTensor
        :acknowledgements: This function was adapted from
        https://github.com/hitomi-team/sukima/blob/main/app/gpt/warpers.py.
        """
        if self.bias is None:
            
            # 1) Create a new tensor that is the same size as the columns of the scores tensor,
            # representing the modified chat history.
            self.bias = torch.zeros(scores.shape[1]).float()
            logit_bias = torch.tensor(self.logit_bias)
            # 2) Fill in the tensor with self.logit_biases to ensure the tokens that are
            # biased get their extra bias value.
            self.bias.scatter_(0, logit_bias[:, 0].long(), logit_bias[:, 1].float())
            # 3) Standardize the tensor such that it's the same dimensions as the input_ids,
            # and convert to the backend's selected device.
            self.bias = self.bias.to(scores.dtype).to(scores.device).unsqueeze(0)
        return scores + self.bias

