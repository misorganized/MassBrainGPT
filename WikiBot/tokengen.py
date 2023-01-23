import string

# Define the characters to include in the tokens
alphabet = string.ascii_letters + string.digits + string.punctuation + "!@#$%^&*_+-"

# Open the file for writing
with open("tokens.txt", "w") as f:
    # Generate all possible 3 letter combinations
    for c1 in alphabet:
        for c2 in alphabet:
            for c3 in alphabet:
                token = c1 + c2 + c3
                # Write the token to the file
                f.write(token + "\n")