import click
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from torchsummary import summary
from dataset import CustomCIFAR10
from model import RepresentativeModel, ContrastiveLoss, FineTunedModel


@click.command()
@click.option('--embedding_dim', default=512, help='Embedding dimension')
@click.option('--batch_size', default=128, help='Batch size')
@click.option('--epochs', default=30, help='Epochs')
@click.option('--max_early_stop', default=5, help='CLR max early stop count')
@click.option('--learning_rate', default=0.001, help='Learning rate')
@click.option('--temperature', default=0.5, help='Temperature')
@click.option('--num_classes', default=10, help='Classes')
def main(embedding_dim, batch_size, epochs, max_early_stop, learning_rate, temperature, num_classes):
    ## Set GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available')


    ## Load CIFAR-10
    train_dataset = DataLoader(CustomCIFAR10(root='./data', train=True), batch_size=batch_size, shuffle=True)
    print(f'Train dataloader size: {train_dataset.__len__()}')

    valid_dataloader = DataLoader(CustomCIFAR10(root='./data', train=False), batch_size=batch_size, shuffle=True)
    print(f'Valid dataloader size: {valid_dataloader.__len__()}')


    ## Show Augmented images
    for data in train_dataset:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow((data[0])[0].permute(1, 2, 0), vmin=0, vmax=255)
        plt.title('Original')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 3, 2)
        plt.imshow((data[1])[0].permute(1, 2, 0), vmin=0, vmax=255)
        plt.title('Augmented 1')
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 3, 3)
        plt.imshow((data[2])[0].permute(1, 2, 0), vmin=0, vmax=255)
        plt.title('Augmented 2')
        plt.xticks([])
        plt.yticks([])

        plt.show()
        plt.close()
        break


    ## Train CLR model
    representative_model = RepresentativeModel(embedding_dim).to(device)
    representative_criterion = ContrastiveLoss(temperature).to(device)
    representative_optimizer = optim.Adam(representative_model.parameters(), lr=learning_rate)

    min_loss = float('inf')
    cnt_early_stop = 0
    representative_model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        pass_cnt = 0
        for batch in train_dataset:
            _, augmented_1, augmented_2, _ = batch

            if len(augmented_1) != batch_size:
                pass_cnt += 1
                continue

            augmented_1 = augmented_1.to(device)
            augmented_2 = augmented_2.to(device)

            representative_optimizer.zero_grad()

            zi = representative_model(augmented_1)
            zj = representative_model(augmented_2)

            loss = representative_criterion(zi, zj)
            loss.backward()

            representative_optimizer.step()
            total_loss += loss.item()

        total_loss = total_loss / (len(train_dataset) - pass_cnt)
        print(f'Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.5f}')

        if round(total_loss, 2) < round(min_loss, 2):
            min_loss = total_loss
            torch.save(representative_model.state_dict(), 'CLR.pth')
            cnt_early_stop = 0
            print('Model save..')
        else:
            cnt_early_stop += 1
            print(f'Early Stop Count: {cnt_early_stop}..')
            if cnt_early_stop == max_early_stop:
                break
    representative_model.load_state_dict(torch.load('CLR.pth'))

    ## Train Fine-tuning model
    finetuning_model = FineTunedModel(embedding_dim, num_classes, representative_model).to(device)
    finetuning_criterion = nn.CrossEntropyLoss().to(device)
    # finetuning_optimizer = optim.SGD(finetuning_model.parameters(), lr=learning_rate, momentum=0.9)
    finetuning_optimizer = optim.Adam(finetuning_model.parameters(), lr=learning_rate)

    summary(finetuning_model, (3, 32, 32))

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    for epoch in range(epochs):
        finetuning_model.train()
        train_loss = 0.0
        train_acc = []
        min_loss = float('inf')

        for batch in train_dataset:
            image, _, _, labels = batch
            image = image.to(device)
            labels = labels.to(device)

            finetuning_optimizer.zero_grad()
            outputs = finetuning_model(image)
            loss = finetuning_criterion(outputs, labels)
            loss.backward()
            finetuning_optimizer.step()
            train_loss += loss.item()

            pred = torch.max(outputs, 1)[1]
            train_acc.append(((labels == pred).sum() / len(labels)).cpu())

        finetuning_model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_acc = []
            for batch in valid_dataloader:
                image, _, _, labels = batch
                image = image.to(device)
                labels = labels.to(device)

                outputs = finetuning_model(image)
                loss = finetuning_criterion(outputs, labels)
                val_loss += loss.item()

                pred = torch.max(outputs, 1)[1]
                val_acc.append(((labels == pred).sum() / len(labels)).cpu())

        train_loss = train_loss / len(train_dataset)
        val_loss = val_loss / len(valid_dataloader)
        train_acc = np.average(train_acc)
        val_acc = np.average(val_acc)
        print(f'Epoch [{epoch + 1}/{epochs}] Train Loss: {train_loss:.5f} | Valid Loss: {val_loss:.5f} | '
              f'Train Acc: {train_acc:.4f} | Valid Acc: {val_acc:.4f}')
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        if val_loss < min_loss:
            torch.save(finetuning_model.state_dict(), 'FT.pth')

    fig, axs = plt.subplots(1, 2)
    loss_ax = axs[0]
    loss_ax.plot(train_loss_history, 'y', label='train loss')
    loss_ax.plot(val_loss_history, 'r', label='val loss')

    acc_ax = axs[1]
    acc_ax.plot(train_acc_history, 'b', label='train acc')
    acc_ax.plot(val_acc_history, 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    acc_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')

    loss_ax.legend(loc='upper right')
    acc_ax.legend(loc='lower right')

    plt.show()

    # ## Test Fine-tuning model
    # finetuning_model.load_state_dict(torch.load('FT.pth'))
    # finetuning_model.eval()
    # test_acc = []
    #
    # with torch.no_grad():
    #     for batch in valid_dataloader:
    #         images, _, labels = batch
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         outputs = finetuning_model(images)
    #         pred = torch.max(outputs, 1)[1]
    #         test_acc.append(((labels == pred).sum() / len(labels)).cpu())
    #
    # accuracy = np.average(test_acc)
    # print(f'Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    main()