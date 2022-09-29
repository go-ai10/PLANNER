# Generate video for a program. Make sure you have the executable open
import sys

sys.path.append('../simulation')
from unity_simulator.comm_unity import UnityCommunication


script = ['<char0> [Walk] <tv> (1)', '<char0> [switchon] <tv> (1)', '<char0> [Find] <controller> (1)','<char0> [Grab] <controller> (1)', '<char0> [Walk] <sofa> (1)', '<char0> [Sit] <sofa> (1)', '<char0> [Watch] <tv> (1)'] # Add here your script

# script = ['[Walk] <kitchen> (1)',
# '[Walk] <fridge> (1)',
# '[Find] <fridge> (1)',
# '[Open] <fridge> (1)',
# '[Find] <food_bread> (1)'
# '[Grab] <food_bread> (1)',
# '[Close] <fridge> (1)',
# '[Walk] <electrical_outlet> (1)',
# '[Find] <toaster> (1)',
# '[PlugIn] <toaster> (1)',
# '[PutBack] <food_bread> (1) <toaster> (1)',
# '[SwitchOn] <electrical_outlet> (1)',
# '[TurnTo] <toaster> (1)',
# '[LookAt] <toaster> (1)',
# '[SwitchOn] <toaster> (1)']


print('Starting Unity...')
comm = UnityCommunication()

print('Starting scene...')
comm.reset()
comm.add_character('Chars/Male1')

print('Generating video...')
comm.render_script(script, recording=True, find_solution=True)

print('Generated, find video in simulation/unity_simulator/output/')
