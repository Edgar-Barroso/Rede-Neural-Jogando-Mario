

import pygame
import random
import torch
from torch import nn
import numpy as np

frames = 600 # velocidade de cada geração


largura = 1000
altura = 500

cor_branca = (255,255,255)
cor_vermelho = (255,0,0)
cor_prata = (192,192,192)
cor_amarelo = (255,255,0)
cor_verde = (0,255,0)
velocidade_mapa = 10


class Planta(pygame.sprite.Sprite):
    def __init__(self,imagem):
        self.imagem = imagem
        self.rect = self.imagem.get_rect()
        self.rect.top, self.rect.left = 250,largura + 1000
    def update(self,superficie):
        superficie.blit(self.imagem,self.rect)
    def mover(self):
        self.rect.move_ip(-velocidade_mapa,0)
    def recriar(self):
        if self.rect.left < -80:
            self.rect.top, self.rect.left = 250,largura+random.randint(1000,1500)

class Cano(pygame.sprite.Sprite):
    def __init__(self,imagem):
        self.imagem = imagem
        self.rect = self.imagem.get_rect()
        self.rect.top, self.rect.left = 320,largura+1500
    def update(self,superficie):
        superficie.blit(self.imagem,self.rect)
    def mover(self):
        self.rect.move_ip(-velocidade_mapa,0)
    def recriar(self,planta):
        if self.rect.left < -200:
            self.rect.top, self.rect.left = 320,planta.rect.left + random.randint(500,700)
            
            
        
class Player(pygame.sprite.Sprite):
    def __init__(self,imagem):
        self.solo = 340
        self.imagem = imagem
        self.rect = self.imagem.get_rect()
        self.rect.top, self.rect.left = (self.solo,100)
        
        
    def mover(self,vx,vy):
        self.rect.move_ip(vx,vy)
    
    def update(self,superficie):
        superficie.blit(self.imagem,self.rect)


class MinhaRede(nn.Module):
  def __init__(self,input_size,hidden_size,output_size):
    super(MinhaRede,self).__init__()
    self.hidden = nn.Linear(input_size,hidden_size)
    self.relu = nn.ReLU()
    self.output = nn.Linear(hidden_size,output_size)

  def forward(self,X):
    hidden = self.relu(self.hidden(X))
    output = self.output(hidden)
    return torch.round(output)



def colisao(player,rect):
    if player.rect.colliderect(rect):
        return True
    else:
        return False

def main(redeneural:MinhaRede):
    pygame.init()
    tela = pygame.display.set_mode([largura,altura])

    relogio = pygame.time.Clock()
    
    
    
    img_mario = pygame.image.load('mario.png').convert_alpha()
    img_fundo1 = pygame.image.load('fundo.png').convert_alpha()
    img_fundo2 = pygame.image.load('fundo.png').convert_alpha()
    img_cano = pygame.image.load('cano.png').convert_alpha()
    img_planta = pygame.image.load('planta.png').convert_alpha()
    jogador = Player(img_mario)
    cano = Cano(img_cano)
    planta = Planta(img_planta)
    

    pontos = 0    
    vx,vy = 0,0
    velocidade = 20
    sair = False
    x=0
    y=largura


    
    while sair is False:
        if pontos >= 1000:
            relogio.tick(60)
        pontos +=1
        planta.recriar()
        cano.recriar(planta)
        
        
        
        y-=velocidade_mapa
        x-=velocidade_mapa
        if x == -largura:
            
            x=largura
        if y == -largura:
            y = largura
        
        
        
        if jogador.rect.top < jogador.solo:
            vy += 1
        elif jogador.rect.top >= jogador.solo:
            vy = 0
            jogador.rect.top = jogador.solo


            
        dist1 = (cano.rect.left - jogador.rect.left)/1000
        dist = torch.Tensor([dist1])
        decisao = redeneural(dist)
        if decisao.data.numpy() >= 0 and jogador.rect.top == jogador.solo:
            vy = - velocidade


        tela.blit(img_fundo1,(x,0))
        tela.blit(img_fundo2, (y,0))
        jogador.update(tela)
        jogador.mover(vx, vy)
        cano.update(tela)

        cano.mover()
        planta.update(tela)
        planta.mover()

        if colisao(jogador,cano):
            pygame.quit()
            return [pontos,redeneural.hidden.weight,redeneural.output.weight]


        relogio.tick(frames)
        pygame.display.update()
    pygame.quit()

def mutacao(tensor:torch.Tensor):
    tensor.requires_grad = False
    for n, c in enumerate(tensor):
        if random.randint(0,100) <= 5:
            c = c+torch.Tensor([random.randint(-1, 1)])
            tensor[n] = c
    tensor.requires_grad = True
    return tensor
    


if __name__ == '__main__':
    input_size = 1
    hidden_size = 4
    output_size = 1
    notas = []
    for c in range(20):
        redeneural = MinhaRede(input_size, hidden_size, output_size)
        notas.append(main(redeneural))
    notas_ordenadas = sorted(notas, key=lambda x:x[0], reverse=True)
    melhor_da_populacao  = notas_ordenadas[0][:]
    notas = []
    j = 0
    while True:
        j+=1
        redeneural = MinhaRede(input_size, hidden_size, output_size)
        redeneural.hidden.weight = mutacao(melhor_da_populacao[1])
        redeneural.output.weight = mutacao(melhor_da_populacao[2])
        notas.append(main(redeneural))
        if j%20 == 0:
            notas_ordenadas = sorted(notas, key=lambda x:x[0], reverse=True)
            melhor_da_populacao  = notas_ordenadas[0][:]
            notas = notas_ordenadas[0:2][:]

