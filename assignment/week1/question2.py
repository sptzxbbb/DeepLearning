import os

def countName(path):
    with os.scandir(path) as it:
        count = 0
        for entry in it:
            if entry.is_dir():
                count += countName(path + '/' + entry.name)
            else:
                print(entry.name)
                count += 1
    return count


if __name__ == "__main__":
    path = './face'
    print(countName(path))

  