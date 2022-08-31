import os
import time
import neat
from pyautogui import locate
from loader import *
from PIL import Image
import pickle
from loguru import logger # For debug


logger.add('debug.log', level='INFO',
    format='{time} {level} - {message}')


def find_obstacle(region_left: int) -> tuple:
    '''Finds closest obstacle (cactus or bird) on canvas and
    returns a tuple (obs_x, obs_w, obs_h) of it's data.'''
    a = []
    for obs in os.listdir('images\\obstacles'):
        obs_path = 'images\\obstacles\\' + obs
        if 'bird' in obs:
            obs_region = (region_left, 90, 720, 175)
        else:
            obs_region = (region_left, 110, 720, 155)

        res = locate(
            obs_path, 
            'canvas.png', 
            region = obs_region,
            confidence = 0.9)

        if res != None:
            obs_x, obs_y = res.left, res.top
            a.append( (obs_x, obs_y, obs_path) )

    if a == []:
        return (1000, 0, 0)

    obs_x, obs_y, obs_path = min(a)
    with Image.open(obs_path) as obs_img:
        obs_w = obs_img.size[0] - 5
    obs_h = 165 - obs_y
    obs_data = (obs_x, obs_w, obs_h)
    return obs_data


def find_dino() -> tuple:
    '''Finds dino on canvas and returns a tuple (x, y) with his position.'''
    dino_box = locate('images\\dino.png', 'canvas.png',
        region = (0, 0, 300, 170), confidence = 0.7)

    if dino_box == None:
        logger.warning('dino is not located on canvas.png')
        return (None, None)

    dino_x =  dino_box.left + dino_box.width
    dino_y = dino_box.top
    return (dino_x, dino_y)


def game_is_over():
    '''Checks if game_over_label is located on canvas'''
    res = locate(
        'images\\game_over_label.png',
        'canvas.png',
        region = (280, 40, 550, 80),
        confidence = 0.8
    )
    if res != None:
        return True


def eval_genomes(genomes, config):
    logger.info('New generation')
    for _, genome in genomes:
        logger.info(f'Training {genome.key} genome')
        press_space()
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        run = True

        canvas.screenshot('canvas.png')
        dino_x = find_dino()[0]
        old_obs_x = 1000
        
        # Run game for genome
        while run:
            # start_time = time.time()
            canvas.screenshot('canvas.png')

            # Lower fitness if dino smashed into an obstacle
            if game_is_over():
                genome.fitness -= 10
                run = False
                time.sleep(0.5)
                logger.info(f'Finished on with fitness {genome.fitness}')
                continue

            obs_x, obs_w, obs_h = find_obstacle(dino_x + 20)
            obs_dist = obs_x - dino_x
            dino_y = find_dino()[1]
    
            # Get responce from genome (jump / not jump)
            output = net.activate([obs_dist, obs_w, obs_h, dino_y]) 
            logger.debug(f'Input: [{obs_dist}, {obs_w}, {obs_h}, {dino_y}]')

            # Lower fitness for any jump
            if output[0] > 0.5:
                genome.fitness -= 1
                press_space()
                logger.debug('Genome jumped')

            # Increace fitness if obstacle is crossed
            if obs_x > old_obs_x:
                genome.fitness += 5
                logger.debug('Genome crossed obstacle')

            old_obs_x = obs_x

            # end_time = time.time()
            # print(end_time - start_time)
            # time.sleep(0.2)
  

if __name__ == '__main__':  
    canvas = get_canvas()

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-feedforward.txt')

    # Create the population
    p = neat.Population(config)

    try:
        # Run the population for 10 generations
        winner_genome = p.run(eval_genomes, 10)
        logger.info(f'Winner genome - {winner_genome.key}')

        # Save result net in the file to test it later
        winner_net = neat.nn.FeedForwardNetwork.create(winner_genome, config)
        with open('net.pkl', 'wb') as f:
            pickle.dump(winner_net, f)

    except Exception as Ex:
        logger.exception('')
    finally:
        driver.quit()
    