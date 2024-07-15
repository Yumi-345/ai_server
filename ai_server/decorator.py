# run = t.run(run)
import time 


class Test():
    def __init__(self) -> None:
        self.index = 0

    def run(self, f):
        def wrapper(*args,**kwargs):
            print(f"In the test {self.index}*****************************")
            self.index += 1
            try:
                f(*args,**kwargs)
                return True
            except Exception as error:
                print("error:", error)
                return False
        return wrapper

t = Test()

@t.run
def run(name):
    print(f"In the RUN {1+name}")


print(run(10))
time.sleep(1)
print(run("xuwj"))
