from typing import TypeVar, Generic

# 定义一个类型变量 T
T = TypeVar('T')

# 定义一个泛型类
class Stack(Generic[T]):
    def __init__(self):
        self.items = []

    def push(self, item: T):
        self.items.append(item)

    def pop(self) -> T:
        if self.items:
            return self.items.pop()
        return None

    def is_empty(self):
        return len(self.items) == 0

# 使用示例
# 创建一个存储整数的栈
int_stack = Stack[int]()
str_stack = Stack[str]()

print(type(int_stack), type(int_stack).__mro__)
print(type(str_stack), type(str_stack).__mro__)