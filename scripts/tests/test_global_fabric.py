
import lightning as L


def main():
    accelerator = "cpu"
    devices = 1

    # Initialize fabric
    global fabric 
    fabric = L.Fabric(accelerator=accelerator, devices=devices)
    fabric.launch()

    some_function()


def some_function():
    # Do something with fabric
    print(fabric)



if __name__ == '__main__':
    main()