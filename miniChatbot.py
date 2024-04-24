from sentence_transformers import SentenceTransformer, util

questions = [
    "What is your name?",
    "How old are you?",
    "Where are you from?",
    "What do you like to do?",
    "What are your hobbies?",
    "What is your favorite food?",
    "What is your favorite color?",
    "What is your favorite movie?",
    "What is your favorite book?",
    "Do you have any pets?"
]

answers = [
    "My name is ChatBot.",
    "I am just a computer program, so I don't have an age.",
    "I exist in the realm of the internet.",
    "I enjoy chatting with people.",
    "My hobbies include helping users and providing information.",
    "I don't eat, but I like talking about food!",
    "I like all colors, but I don't have a favorite.",
    "I don't watch movies, but I can recommend some!",
    "I don't read books, but I can suggest some good ones!",
    "No, I don't have any pets. I'm just a program."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_highest_similarity(user_input, questions):
    max_similarity = 0
    best_question = None
    for question in questions:
        similarity = util.cos_sim(
            model.encode(user_input),
            model.encode(question)
        )[0][0].item()
        if similarity > max_similarity:
            max_similarity = similarity
            best_question = question
    return best_question, max_similarity

def chatbot(user_input):
    if user_input in questions:
        idx = questions.index(user_input)
        print(answers[idx])
    else:
        best_question, similarity = get_highest_similarity(user_input, questions)
        if similarity >= 0.7:
            idx = questions.index(best_question)
            print(answers[idx])
        else:
            print("Sorry, no answer.")

if __name__ == "__main__":
    print("Welcome to the chatbot! You can ask me anything.")
    print("Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("ChatBot: Goodbye!")
            break
        chatbot(user_input)