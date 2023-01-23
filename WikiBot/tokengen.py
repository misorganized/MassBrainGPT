import string

# Define the characters to include in the tokens
alphabet = string.ascii_letters + string.digits + string.punctuation + "!@#$%^&*_+-\n "

with open("tokens.txt", "w") as f:
    for c1 in alphabet:
        for c2 in alphabet:
            for c3 in alphabet:
                token = c1 + c2 + c3
                f.write(token + "\n")