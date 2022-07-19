



class LocalMemory:
    def __init__(self) -> None:
        self.memory = []

    def get(self, index):
        assert len(self.memory) > 0
        return self.memory[index]
    
    def append(self, data):
        self.memory.append(data)

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)