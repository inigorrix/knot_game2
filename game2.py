import pygame
import numpy as np
import random

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg

from diagram import Diagram

from knot_functions import clean_k_arc, xco_arc, arc_xoco, xabo_xco
from knot_functions import circles_crd_cds, random_knot


pygame.init()

width = 960
aspect_ratio = 9/16
winsize = (width, round(width*aspect_ratio))
win = pygame.display.set_mode(winsize, pygame.RESIZABLE)
pygame.display.set_caption('Game')

pygame.scrap.init()

text_rect = pygame.Rect(0,0,0,0)
text_active = False
user_text = ''
font_name = None
base_font = pygame.font.SysFont(font_name, 0)

timer_rect = pygame.Rect(0,0,0,0)
timer_running = False
timer_edit = False
timer_input = ''
timer_time = 30
timer_left = timer_time
timer_start = 0

bg_color = (125, 125, 125)
white = (255, 255, 255)
black = (0, 0, 0)
red = (125, 25, 25)

arc = Diagram()
surface = Diagram()
a_circles = Diagram()
b_circles = Diagram()

show_circles = True

game_on = False
game_history = []

btn_text = ['-', '=', '+', '\u292b', '\u292c',
            '\u25cb', '\u2731', '\u293e']
knot_buttons = [None]*len(btn_text)


xco_i = xco_arc(clean_k_arc('[[5,2],[1,3],[2,4],[3,5],[4,1]]'))


def new_knot(i):
    if i==0:
        Diagram.n -= 1
    elif i==2:
        Diagram.n += 1

    if Diagram.n>4:
        define_knot(random_knot(Diagram.n))
    else:
        Diagram.n+=1

def define_knot(xco):
    global k_xco, cross_rc, ab_crossings
    k_xco = xco
    cross_rc = np.argwhere(k_xco==3)
    ab_crossings=np.array([random.randint(0,1) for _ in range(len(cross_rc))])

    arc.changeable_crossings = [list(rc) for rc in cross_rc]

    update_knots()

def update_knots():
    arc.xabo = xabo_xco(k_xco, 4+ab_crossings)
    surface.xabo = arc.xabo.copy()

    a_circles.xabo = xabo_xco(k_xco, 4+ab_crossings)
    b_circles.xabo = xabo_xco(k_xco, 5-ab_crossings)

    coords_u, codes_u = circles_crd_cds(xabo_xco(k_xco, 4+ab_crossings))
    coords_d, codes_d = circles_crd_cds(xabo_xco(k_xco, 5-ab_crossings))
    surface.n_circles_ud[:] = len(codes_u), len(codes_d)

    Diagram.n = k_xco.shape[0]

    window_resized()

def resize_grids(winsize):
    g_size, g_margin = winsize[0]/3, winsize[0]/24
    g_coord_y = (winsize[1] - g_size)/2
    
    arc.grid_size = g_size
    arc.sqr_size = g_size/Diagram.n
    arc.grid_coords = (winsize[0]-(g_size+g_margin), g_coord_y)
    
    surface.grid_coords = (g_margin, g_margin)
    surface.grid_size = g_coord_y + g_size - g_margin

    surf_r_coord = g_margin + surface.grid_size
    margin = (arc.grid_coords[0] - surf_r_coord) * .1
    c_size = (arc.grid_coords[0] - surf_r_coord) - 2*margin
    a_circles.grid_size, b_circles.grid_size = c_size, c_size

    coord_x = surf_r_coord + margin
    coord_ya = surface.grid_coords[1] + margin
    coord_yb = surface.grid_coords[1] + surface.grid_size - (margin + c_size)
    a_circles.grid_coords = (coord_x, coord_ya)
    b_circles.grid_coords = (coord_x, coord_yb)
    

def resize_buttons(winsize):
    n = len(knot_buttons)
    size = arc.grid_size/n
    x0, y = arc.grid_coords[0], arc.grid_coords[1]-size
    for i in range(n):
        x_i = x0 + size*i
        knot_buttons[i] = pygame.Rect(x_i, y, size, size)

def draw_buttons():
    k_b_border = int(np.ceil(knot_buttons[0].size[0]*.02))
    n = len(knot_buttons)
    btn_size = arc.grid_size/n
    btn_font = pygame.font.Font("segoe-ui-symbol.ttf",int(btn_size))
    for i in range(n):
        btn_rect = knot_buttons[i]
        pygame.draw.rect(win, white, btn_rect)
        pygame.draw.rect(win, bg_color, btn_rect, k_b_border)

        #buttons text
        text_surface = btn_font.render(btn_text[i], True, black)
        if i==3 or i==4:
            text_surface = pygame.transform.rotate(text_surface, 45)
        text_w, text_h = text_surface.get_rect().size
        text_pos = btn_rect.centerx-text_w*.525, btn_rect.centery-text_h*.525
        win.blit(text_surface, text_pos)
        
def draw_timer():
    pygame.draw.rect(win, white, timer_rect)
    timer_border = round(pygame.display.get_window_size()[1] * 0.01)
    col = black if timer_edit else white
    pygame.draw.rect(win, col, timer_rect, timer_border)

    timertext=str(int(timer_left)) if timer_left>1 else str(round(timer_left,1))
    if timer_edit:
        timertext = timer_input
    text_surface = base_font.render(timertext+' s', True, black)
    text_w, text_h = text_surface.get_rect().size
    text_pos = timer_rect.centerx-text_w*.5, timer_rect.centery-text_h*.5
    win.blit(text_surface, text_pos)

def draw_texts():
    pygame.draw.rect(win, white, text_rect)
    text_border = round(pygame.display.get_window_size()[1] * 0.01)
    col = black if text_active else bg_color
    pygame.draw.rect(win, col, text_rect, text_border)

    text_surface = base_font.render(user_text, True, black)
    text_w, text_h = text_surface.get_rect().size
    text_pos = text_rect.centerx-text_w*.5, text_rect.centery-text_h*.5
    win.blit(text_surface, text_pos)
    
    k_coords = str(arc_xoco(k_xco)).replace('\n', '')
    text_surface = base_font.render(k_coords, True, black)
    text_w, text_h = text_surface.get_rect().size
    text_posx = text_rect.centerx-text_w*.5
    text_posy = text_rect.centery-text_h*.5-text_rect.h
    win.blit(text_surface, (text_posx, text_posy))

    if show_circles:
        gc_x, gc_ya = a_circles.grid_coords
        gc_yb, gs = b_circles.grid_coords[1], b_circles.grid_size
        n_c = surface.n_circles_ud

        text_surface = base_font.render('|SaD|='+str(n_c[0]), True, black)
        text_w, text_h = text_surface.get_rect().size
        text_pos = gc_x, gc_ya - text_h
        # Show number of circles UP
        win.blit(text_surface, text_pos)

        text_surface = base_font.render('|SbD|='+str(n_c[1]), True, black)
        text_w, text_h = text_surface.get_rect().size
        text_pos = gc_x, gc_yb + gs
        # Show number of circles DOWN
        win.blit(text_surface, text_pos)


def draw_window():
    win.fill(bg_color)

    arc.draw_grid(win)

    if game_on:
        arc.color_crossings(win)
        pygame.draw.rect(win, black,
                         (arc.grid_coords,(arc.grid_size,arc.grid_size)),
                         int(arc.grid_size*.01))
    
    arc.draw_arc_mod(win)

    surface.draw_surface(win)
    if show_circles:
        a_circles.draw_ab_circles(win)
        b_circles.draw_ab_circles(win)

    draw_texts()
    draw_buttons()
    draw_timer()
    
    pygame.display.update()


def window_resized():
    global base_font

    winwidth = pygame.display.get_window_size()[0]
    if winwidth != pygame.display.get_desktop_sizes()[0][0]:
        winsize = (winwidth, winwidth*aspect_ratio)
        pygame.display.set_mode(winsize, pygame.RESIZABLE)

    winsize = pygame.display.get_window_size()
    resize_grids(winsize)
    resize_buttons(winsize)

    text_rect.size = (winsize[0], winsize[1]*.08)
    text_rect.topleft = (0, winsize[1]-text_rect.size[1])

    base_font = pygame.font.SysFont(font_name, int(text_rect.size[1]*.8))

    a_gc, b_gc = a_circles.grid_coords, b_circles.grid_coords
    gs = a_circles.grid_size
    t_space_y = b_gc[1] - (a_gc[1]+gs)
    timer_rect.size = gs, t_space_y * .8
    timer_rect.center = a_gc[0] + gs/2, (a_gc[1]+gs + b_gc[1])/2
    
    
def mouse_clicked(pos):
    global text_active, timer_edit, ab_crossings, show_circles
    global game_on, game_history
    
    if text_rect.collidepoint(pos):
        text_active = True
    else:
        text_active = False

    if timer_rect.collidepoint(pos) and not(timer_running):
        timer_edit = True
    else:
        timer_edit = False

    arc_gs = arc.grid_size
    arc_rect = pygame.Rect(arc.grid_coords, (arc_gs,arc_gs))
    if arc_rect.collidepoint(pos):
        click_rc = arc.grid_r_c(pos)

        if (k_xco[click_rc]==3 and
            (arc.active_crossing == click_rc or not(game_on))
            ):
            click_cross_n = np.where(np.all(cross_rc==click_rc, axis=1))[0][0]
            ab_crossings[click_cross_n] = (ab_crossings[click_cross_n]+1)%2
            update_knots()

        if k_xco[click_rc]==3 and list(click_rc) in arc.changeable_crossings:
            if arc.active_crossing == None and game_on:
                arc.active_crossing = click_rc
                game_history.append([ab_crossings.copy(),
                                     arc.changeable_crossings.copy(),
                                     None])


    for i in range(len(knot_buttons)):
        if knot_buttons[i].collidepoint(pos):
            if not(game_on):
                if i<3:
                    new_knot(i)
                elif i==3:
                    ab_crossings=np.zeros(cross_rc.shape[0])
                elif i==4:
                    ab_crossings=np.ones(cross_rc.shape[0])

            if i==5:
                    show_circles = not(show_circles)
            elif i==6:
                game_on = not(game_on)
                if game_on:
                    arc.active_crossing = None
                    arc.changeable_crossings = [list(rc) for rc in cross_rc]
                    game_history = []
            elif i==7:
                if game_on and len(game_history)>0:
                    ab_crossings = game_history[-1][0].copy()
                    arc.changeable_crossings = game_history[-1][1].copy()
                    arc.active_crossing = game_history[-1][2]
                    game_history.pop(-1)
                
            update_knots()

def key_pressed(event):
    global user_text, show_circles, timer_time, timer_left
    global timer_start, timer_running, timer_edit, timer_input
        
    if text_active or timer_edit:
        input_text = user_text if text_active else timer_input
        
        if event.key == pygame.K_BACKSPACE:
            input_text = user_text[:-1]
        elif event.key == pygame.K_DELETE:
            input_text = ''
        else:
            input_text += event.unicode

        if text_active:
            user_text = input_text
            if event.mod & pygame.KMOD_CTRL and event.key==pygame.K_v:
                clipboard = pygame.scrap.get(pygame.SCRAP_TEXT)
                user_text += clipboard.decode('utf-8')[:-1]
            elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                if user_text!='':
                    k_arc = clean_k_arc(user_text)
                    if type(k_arc)!=type(None):
                        define_knot(xco_arc(k_arc))
                        user_text = ''
        elif timer_edit:
            timer_input = input_text
            if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                if timer_input!='':
                    try:
                        timer_time = int(timer_input)
                        timer_left = timer_time
                        timer_edit = False
                    except Exception:
                        pass
                    timer_input = ''

    elif event.key == pygame.K_SPACE:
        timer_running = not(timer_running)
        if timer_running:
            timer_start = pygame.time.get_ticks()-(timer_time-timer_left)*1000
            timer_edit = False
            show_circles = False
            
            
    elif event.key == pygame.K_m:
        surface.zoom /= .75
    elif event.key == pygame.K_n:
        surface.zoom *= .75
        if surface.zoom<1: surface.zoom /= .75
    elif event.key == pygame.K_v:
        surface.view = arc.view.copy()
        surface.zoom = arc.zoom
    elif event.key == pygame.K_s:
        #print('\nVIEW\nazim:', surface.view[0], '\telev:', surface.view[1])
        #print('\nWINDOW SIZE:', pygame.display.get_window_size())
        print(game_history)
        print()

    elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
        if arc.active_crossing != None and game_on:
            arc.changeable_crossings.remove(list(arc.active_crossing))
            game_history.append([ab_crossings.copy(),
                                 game_history[-1][1].copy(),
                                 arc.active_crossing])
            arc.active_crossing = None


def update_timer():
    global timer_left, timer_running, show_circles

    timer_left = timer_time-(pygame.time.get_ticks()-timer_start)/1000

    if timer_left<0:
        timer_running = False
        timer_left = timer_time
        show_circles = True


def main():
    define_knot(xco_i)
    
    run = True
    while run:
        draw_window()
        
        keys_p = pygame.key.get_pressed() # keys_pressed
        
        if timer_running:
            update_timer()
            events = pygame.event.get()
            
        elif any([keys_p[pygame.K_UP], keys_p[pygame.K_DOWN],
               keys_p[pygame.K_LEFT], keys_p[pygame.K_RIGHT]]):
            if keys_p[pygame.K_UP]:
                surface.view[1] += 5
            if keys_p[pygame.K_DOWN]:
                surface.view[1] -= 5
            if keys_p[pygame.K_RIGHT]:
                surface.view[0] += 5
            if keys_p[pygame.K_LEFT]:
                surface.view[0] -= 5

            if keys_p[pygame.K_LSHIFT] or keys_p[pygame.K_RSHIFT]:
                events = [pygame.event.wait()] + pygame.event.get()
            else:
                events = pygame.event.get()
            
        else:
            events = [pygame.event.wait()]+pygame.event.get()

        for event in events:
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button==1:
                    mouse_clicked(event.pos)
                    
            elif event.type == pygame.KEYDOWN:
                pygame.key.set_repeat(0)
                key_pressed(event)
                        
            elif event.type == pygame.WINDOWRESIZED:
                window_resized()

            elif event.type == pygame.QUIT:
                run = False
                
                
    pygame.quit()


if __name__=='__main__':
    main()
