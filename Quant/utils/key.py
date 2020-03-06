def get_key():
    with open('key.txt', 'rb') as k:
        return str(k.readline())
if __name__ == "__main__":
    get_key()