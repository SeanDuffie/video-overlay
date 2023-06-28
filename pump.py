import logging
import sys
import time

import keyboard
import serial
import serial.tools.list_ports as port_list


class SyringePump:
    def __init__(self, port: str = "") -> None:
        fmt_pump = "%(asctime)s | %(levelname)s |\tPump:\t%(message)s"
        logging.basicConfig(format=fmt_pump, level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S")

        if port == "":
            ports = list(port_list.comports())
            for i, p in enumerate(ports):
                print(i, ") ", p.name)
            sel = int(keyboard.read_key())
            port: str = ports[sel].name
        self.ser = serial.Serial(
            port=port,
            baudrate=115200)

    ### Command List
    # - cat - prints a list of custom presets that have been made on the pump
    # - config - modifies configuration, TODO: I need to look into this later
    # - echo - sets/gets whether the serial connection should respond with the inputs it receives
    # - force* - sets/displays [1-100] the current infusion force level
    # - load* - loads/displays the current preset method (from cat)
    # - irun* - runs in infuse direction
    # - wrun* - runs in withdraw direction
    # - rrun - runs in reverse?? direction
    # - run* - simulates pressing the run button
    # - stp* - stops the pump
    # - crate* - displays the current rate and direction of the motor
    # - diameter - sets/displays the syringe diameter in mm
    # - {i,w}ramp* - sets/displays the infusion rates while ramping
    # - {i,w}rate/wrate* - sets/displays the infusion rate
    # - {ci,ctc,cw,i,s,t,w}volume - dealing with the measured volume
    # - {ci,ctc,cw,i,s,t,w}time - dealing with the measured time
    def load(self, method: str = "") -> None:
        # Create message to send
        msg = "load"
        if method == "":
            msg += "\r"
        else:
            msg += f" {method}\r"

        # Serial Interaction
        self.ser.write(bytes(msg.encode()))
        time.sleep(.1)
        if method == "":
            res = ""
            while self.ser.inWaiting() > 1:
                res += self.ser.readline().decode() + "\n"
            logging.info(res)


    def run(self):
        # Serial Interaction
        self.ser.write(bytes("run\r".encode()))
        time.sleep(.1)

    def stop(self):
        # Serial Interaction
        self.ser.write(bytes("stp\r".encode()))
        time.sleep(.1)

    def interactive(self):
        try:
            while True:
                Key = keyboard.read_key()
                if Key == "1":
                    msg = "cat\r"
                elif Key == "2":
                    msg = "echo\r"
                elif Key == "3":
                    msg = "run\r"
                elif Key == "4":
                    msg = "stp\r"
                elif Key == "5":
                    msg = "load\r"
                elif Key == "6":
                    msg = "load PRGM_TEST\r"
                elif Key == "7":
                    msg = "load qs i\r"
                else:
                    logging.warning("Unknown Keypress: %s", Key)
                    msg = ""
                logging.info("Command: %s", msg)
                self.ser.write(bytes(msg.encode()))
                time.sleep(1)
                res = ""
                while self.ser.inWaiting() > 1:
                    res += self.ser.readline().decode() + "\n"
                logging.info(res)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    sp = SyringePump(port="COM15")
    # sp.interactive()
    print("loading...")
    sp.load(method="1234")
    print("running...")
    sp.run()
    time.sleep(25)
    print("stopping...")
    sp.stop()
