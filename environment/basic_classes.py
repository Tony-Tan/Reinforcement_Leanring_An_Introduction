class Space:
    def __init__(self, initial_list):
        self.list = initial_list
        self.n = len(initial_list)

    def __getitem__(self, index):
        return self.list[index]