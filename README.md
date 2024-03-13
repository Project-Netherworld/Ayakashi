# Ayakashi
The Backend for Project Netherworld. For more information about the Project, see [this homepage](https://github.com/Project-Netherworld).

## Features 
- Efficient Communication  via Serialization of Tensors and Chat History(using FastAPI and Uvicorn) to Interface with the [Phantasmagoria Front End](https://github.com/Project-Netherworld/Phantasmagoria).
- Implementation of Experimental Warpers such as [top_a](https://github.com/BlinkDL/RWKV-LM/tree/4cb363e5aa31978d801a47bc89d28e927ab6912e#the-top-a-sampling-method) and [Tail Free Sampling](https://www.trentonbricken.com/Tail-Free-Sampling/) to allow more options for creatively generating text via the conversational agent (chatbot)  
- Implementation of Experimental Processors such as [Logit_bias(es)](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api) to customize their conversation agent to say certain words/phrases more or less often
- Quick and Efficient text generation via the Transformers and Pytorch libraries 
- Customizable Port Hosting

## Installation 
0. Have Python installed. **This project requires Python 3.10 or higher.**
1. Either clone this repository, download it via zip, or download the release. 
2. Using your favorite CLI (command prompt, bash, etc.), use `cd` to change the directory to where you downloaded the repository.
3.  Run the following command: 
`pip install -r requirements.txt`
4. Alternatively, should you prefer not using pip and want to use conda instead, run the following command: 
`conda install --file requirements.txt`

## Usage
There is not much to go over the backend, as the user all you are concerned with is setting it up, whether it be locally or on a virtual machine. **If you are hosting non-locally, be sure that you have port forwarded a specific port to ping. Otherwise, this project will not work.**

### Running the Program
1. **To run the program, you will need to run the following command:**
`python backend_server.py`. Some installations of Python 3 utilize `python3` as a prefix to run python commands instead, so if this is the case, run the command as such instead:
`python3 backend_server.py”`
**By default, this will make the backend server run on port 8000.**
2.  Optionally, you can set the port to a custom value simply by passing a number to the run command shown above. For example, you can run the following command to set the backend server to run on port 8080: 
`python backend_server.py 8080.`
3. **You will be aware that the server works if you see the following (or similar) text outputted in your CLI or console:**

![An image showing the Uvicorn/Backend server setup. The most important message here is: "Uvicorn running on http://0.0.0.0:8000 where 8000 can be whatever port number you set it to.](https://raw.githubusercontent.com/Project-Netherworld/.github/main/images/image29.png)
