# client.py
import grpc
import prompt_service_pb2
import prompt_service_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = prompt_service_pb2_grpc.PromptServiceStub(channel)
    
    # Replace this with your desired text prompt
    prompt = "Kumusta , gusto mo bang marinig ang tungkol sa aming bagong pag-aalok ng pautang na higit pa sa mga rate ng merkado"

    response = stub.ProcessPrompt(prompt_service_pb2.PromptRequest(prompt=prompt))

    if response.success:
        print("Audio generation succeeded. Message:", response.message)
        #print("Generated audio saved at:", response.audio_path)
    else:
        print("Audio generation failed. Message:", response.message)

if __name__ == '__main__':
    run()
