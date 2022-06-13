import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

with open("shared_ark_tweebank.pkl","rb") as file:
    shared_examples = pickle.load(file)

def make_sent_vecs(dataset):
    "Given a dataset return sent to vec dataset for model use"
    vec_dataset = [[]]
    sent_transformer = SentenceTransformer('all-mpnet-base-v2')
    for i,data in tqdm(enumerate(dataset),desc="Vectorizing sentences"):
        vec_dataset[-1].append(torch.from_numpy(sent_transformer.encode(data[0])))
        vec_dataset[-1].append(dataset[i][1])
        vec_dataset[-1].append(dataset[i][2])
        vec_dataset.append([])
    return vec_dataset
vec_dataset = make_sent_vecs(shared_examples)
# print(vec_dataset[0][0])
# print(type(vec_dataset[0][0]))
# print(vec_dataset[0][0].shape)



class Encoder(nn.Module):
    '''
    Encoder for label to label model

    '''
    def __init__(self, input_dim, encoder_hidden_dim):
        super(Encoder, self).__init__()
        self.encoderDropout = .2
        self.embedding = nn.Sequential(nn.Linear(input_dim,encoder_hidden_dim//2),
                                       nn.BatchNorm1d(encoder_hidden_dim//2),nn.GELU(),nn.Dropout(self.encoderDropout),
                                       nn.Linear(encoder_hidden_dim//2,encoder_hidden_dim),
                                       nn.BatchNorm1d(encoder_hidden_dim),nn.GELU()
                                       )
        
        

    def forward(self, x):
        """"""
        return self.embedding(x)


print("--------------Encoder Test------------------------")
hidden_dim = 200
smallEncoder = Encoder(768,hidden_dim)
EncoderOutput = smallEncoder.forward(vec_dataset[0][0])
print("Shape of the Encoders Output: ", EncoderOutput.shape)
print("All Encoder Tests Passed!")
print("\n")


class Decoder_YtoZ(nn.Module):
    '''
    Decoder for label to label model
    Given X Encoding returns decoding prediction Zbar
    Given X and decoding for ybar , returns Zbar

    '''
    def __init__(self, input_dim, decoder_hidden_dim,y_classes,z_classes):
        super(Decoder_YtoZ, self).__init__()
        self.decoderDropout = .1
        self.decoding = nn.Sequential(nn.Linear(input_dim,decoder_hidden_dim//2),
                                       nn.BatchNorm1d(decoder_hidden_dim//2),nn.GELU(),
                                       nn.Dropout(self.decoderDropout),
                                       nn.Linear(decoder_hidden_dim//2,decoder_hidden_dim//4),
                                       nn.BatchNorm1d(decoder_hidden_dim//4),nn.GELU(),
                                       nn.Linear(decoder_hidden_dim//4,z_classes)
                                       )
        self.transformYtoZ = nn.Linear(y_classes,z_classes)


    def forward(self, encoded, ybar=None):
        """"""
        if ybar is None:
            decoded = self.decoding(encoded)
        else:
            decoded = self.decoding(encoded)
            transform = self.transformYtoZ(ybar)
            decoded = decoded + transform
        return decoded

print("--------------Decoder Y to Z Test------------------------")
y_classes = 10 #Number of classes for Y 
z_classes = 15 # NUmber of classes for Z
decoder_hidden_dim = 400
smallDecoderyz= Decoder_YtoZ(hidden_dim,decoder_hidden_dim,y_classes,z_classes)
DecoderOutput = smallDecoderyz.forward(smallEncoder.forward(vec_dataset[0][0]))
print("Shape of the Decoders YZ Output: ", DecoderOutput.shape)
print("\n")



class Decoder_ZtoY(nn.Module):
    '''
    Decoder for label to label model
    Given X Encoding returns decoding prediction Ybar
    Given X and decoding for zbar , returns Ybar

    '''
    def __init__(self, input_dim, decoder_hidden_dim,y_classes,z_classes):
        super(Decoder_ZtoY, self).__init__()
        self.decoderDropout = .1
        self.decoding = nn.Sequential(nn.Linear(input_dim,decoder_hidden_dim//2),
                                       nn.BatchNorm1d(decoder_hidden_dim//2),nn.GELU(),
                                       nn.Dropout(self.decoderDropout),
                                       nn.Linear(decoder_hidden_dim//2,decoder_hidden_dim//4),
                                       nn.BatchNorm1d(decoder_hidden_dim//4),nn.GELU(),
                                       nn.Linear(decoder_hidden_dim//4,y_classes)
                                       )
        self.transformZtoY = nn.Linear(z_classes,y_classes)
        

    def forward(self, encoded, zbar=None):
        """"""
        if zbar is None:
            decoded = self.decoding(encoded)
        else:
            decoded = self.decoding(encoded)
            transform = self.transformZtoY(zbar)
            decoded = decoded + transform
        return decoded
print("--------------Decoder Y to Z YBAR Test------------------------")
smallDecoderzy= Decoder_ZtoY(hidden_dim,decoder_hidden_dim,y_classes,z_classes)
DecoderOutput = smallDecoderyz.forward(smallEncoder.forward(vec_dataset[0][0]),ybar=smallDecoderzy.forward(smallEncoder.forward(vec_dataset[0][0])))
print("Shape of the Decoders YZ Output with ybar: ", DecoderOutput.shape)
print("All Decoder YZ Tests Passed!")
print("\n")



print("--------------Decoder Z to Y Test------------------------")
smallDecoderzy= Decoder_ZtoY(hidden_dim,decoder_hidden_dim,y_classes,z_classes)
DecoderOutput = smallDecoderzy.forward(smallEncoder.forward(vec_dataset[0][0]))
print("Shape of the Decoders ZY Output: ", DecoderOutput.shape)
print("\n")

print("--------------Decoder Z to Y ZBAR Test------------------------")
print("test e shape: ", smallEncoder.forward(vec_dataset[0][0]).shape)
print("test decoder shape: ", smallDecoderyz.forward(smallEncoder.forward(vec_dataset[0][0])).shape)
DecoderOutput = smallDecoderzy.forward(smallEncoder.forward(vec_dataset[0][0]),zbar=smallDecoderyz.forward(smallEncoder.forward(vec_dataset[0][0])))
print("Shape of the Decoders ZY Output with zbar: ", DecoderOutput.shape)
print("All Decoder ZY Tests Passed!")
print("\n")

class Label2Label(nn.Module):
    '''
    We train an end-to-end label to label model comprising of Encoder and Two Decoders.
    This is a wrapper "model" for the encoders and decoders.
    '''
    def __init__(self, input_dim, encoder_hidden_dim, decoder_hidden_dim, y_classes,z_classes):
        super(Label2Label,self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim)
        self.decoderYZ = Decoder_YtoZ(encoder_hidden_dim, decoder_hidden_dim,y_classes,z_classes)
        self.decoderZY = Decoder_ZtoY(encoder_hidden_dim, decoder_hidden_dim,y_classes,z_classes)
    def forward(self, x,decode_y=False):
        if decode_y:
            e = self.encoder(x)
            zbar = self.decoderYZ(e)
            out = self.decoderZY(e,zbar)
        else:#Decodes z
            e = self.encoder(x)
            ybar = self.decoderZY(e)
            out = self.decoderYZ(e,ybar)
        return out
input_dim = 768
Label2LabelModel = Label2Label(input_dim, hidden_dim, decoder_hidden_dim, y_classes,z_classes)
print("--------------Label 2 Label Model Test------------------------")
decoding_z = Label2LabelModel.forward(vec_dataset[0][0])
decoding_y = Label2LabelModel.forward(vec_dataset[0][0],decode_y=True)
print("Decoded z: ", decoding_z.shape)
print("Decoded y: ", decoding_y.shape)
print("All tests passed, generic label 2 label model created!")