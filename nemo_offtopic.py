from nemoguardrails import  RailsConfig,LLMRails
config_path = RailsConfig.from_path("./config")
rails = LLMRails(config_path)
quest = input("Enter question\n")
while (quest!='0'):
    response = rails.generate(messages=[{
        "role": "user",
        "content": quest
    }])
    print("Response here")
    print(response["content"])
    if (response["content"]=="True"):
        print("Proceed with Copilot pipeline")
    else:
        print("Understood! But as an FDC copilot, I'm equipped to assist with FDC-related inquiries exclusively. Let me know how I can help within that scope")
    quest = input("Enter question\n")


