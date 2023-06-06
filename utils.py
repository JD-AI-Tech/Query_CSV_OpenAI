
def read_text_file(file_path):
    """Reads a text file on Windows and returns its contents."""
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
        return contents
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except PermissionError:
        print(f"Permission denied to open file '{file_path}'.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

# Example usage:
if __name__ == "__main__":
    file_path = './data/pokemon.csv'
    file_contents = read_text_file(file_path)
    if file_contents:
        print(file_contents)
