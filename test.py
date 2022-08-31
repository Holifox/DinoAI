from train import *


with open('net.pkl', 'rb') as f:
    net = pickle.load(f)

canvas = get_canvas()
press_space()
canvas.screenshot('canvas.png')
dino_x = locate_dino()[0]
run = True

while run:
    canvas.screenshot('canvas.png')

    if game_is_over():
        time.sleep(0.4)
        run = False
        driver.quit()
        exit()

    obs_x, obs_w, obs_h = find_obstacle(dino_x + 20)
    dino_y = locate_dino()[1]

    if obs_x != 1000:
        obs_dist = obs_x - dino_x
    else:
        obs_dist = 1000

    # Get responce from genome (jump / not jump)
    output = net.activate([obs_dist, obs_w, obs_h, dino_y]) 
    
    if output[0] > 0.5:
        press_space()