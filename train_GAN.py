from data_loader import EEGFeatureDataset

from GANs.DC_WGAN_GP import WGAN_GP
#from GANs.DC_WC_GAN import WCDCGAN
#from GANs.DC_GAN import DCGAN
#from GANs.GAN import GAN
#from GANs.WGAN import WGAN


from torch.utils.data import DataLoader

data_dir = '/Users/umairarshad/SHU/or_data/mat'
feature_dir = '/Users/umairarshad/SHU/or_data/csp_features'  

subjects = range(1, 21)  
sessions = range(1,5)  
dataset = EEGFeatureDataset(data_dir, feature_dir, subjects, sessions)
print(dataset.__len__())

batch_size = 100
noise_dim = 100
channels = 3
feature_dim = 3

data_loader = DataLoader(dataset, batch_size, shuffle=True)
print(f"Total number of batches: {len(data_loader)}")





dc_wgan_gp = WGAN_GP(batch_size=100, noise_dim=1000 ,channels=3 , feature_dim=3)
#wcdcgan = WCDCGAN(batchsize=50, noise_dim=100, feature_dim=3)
#dcgan = DCGAN(noise_dim=100, feature_dim=3)  
#wgan = WGAN(batchsize=100, channels=3, noise_dim=1000, feature_dim=feature_dim)
#gan = GAN(noise_dim=50, feature_dim=3)


# Train GAN
epochs = 501

#dc_wgan_gp_update.train(data_loader, epochs)
dc_wgan_gp.train(data_loader, epochs)
#wcdcgan.train(data_loader, epochs)
#dcgan.train(data_loader, epochs)
#wgan.train(data_loader, epochs)
#gan.train(data_loader, epochs)
#gan_new.train(data_loader, epochs)
