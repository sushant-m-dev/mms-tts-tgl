# client.py
import grpc
import prompt_service_pb2
import prompt_service_pb2_grpc

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = prompt_service_pb2_grpc.PromptServiceStub(channel)
    
    while True:
        prompt = input("Enter a text prompt (or type 'exit' to quit): ")

        if prompt.lower() == 'exit':
            break

        response = stub.ProcessPrompt(prompt_service_pb2.PromptRequest(prompt=prompt))

        if response.success:
            print("Audio generation succeeded. Message:", response.message)
        else:
            print("Audio generation failed. Message:", response.message)

if __name__ == '__main__':
    run()
