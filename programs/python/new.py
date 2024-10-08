import random
import time
import linecache

Urn_Main = []       # тут буде urna
Urn_Txt = []        # tyt text

Tokens_txt ={}      # was in urn_txt
Out_Tokens = {}
Sorted_Tokens = {}

def Urn_main_CREATE(urn_main, start_size):        # Створюю робочу урну of selected volume
    Dictionary = open('Dictionary.txt', 'r')      # Відкрили словник
    for i in range(0, start_size):                # В циклі скільки нам треба раз
        word = Dictionary.readline().strip("\n")  # Берем слово під номером лінії і, видаляємо \n
        urn_main.append(word)                     # Додаєм в робочу урну
    Dictionary.close()                            # Закрили по фен-шую словник

def main(urn_main,urn_txt,text_len,start_size,Ro,Nu):
    start_time = time.time()

    Dictionary = open('Dictionary.txt', 'r')  # txt file
    dictionary_counter = start_size+1  # for choosing frm txt | start_size because of Urn_main_CREATE!!!

    line_count = -1
    if text_len > 50000:
        line_count = 0
        for line in Dictionary:
            if line != "\n":
                line_count += 1


    for iterable in range(text_len):
        random_element = random.choice(urn_main)        #!!!!!!!!!!!!!!!!!
        urn_txt.append(random_element)                  #!!!!!!!!!!!!!!!!!

        if random_element in Tokens_txt:                                    # If word is in Tokens dict
            Tokens_txt[random_element] += Ro

            for k in range(Ro):                               # Ro times
                urn_main.append(random_element)               # add same element

        else:                                                                                    #if not in tokens
            Tokens_txt[random_element] = Ro+1

            for k in range(Ro):                                                                     # add ro copies of word
                urn_main.append(random_element)
            for j in range(Nu + 1):                                                         # add nu+1 innovations
                new_word = linecache.getline('Dictionary.txt', dictionary_counter).strip("\n")              # select new word
                dictionary_counter += 1                                                     # dictionary iteration update
                urn_main.append(new_word)                                                   # add Innova to urn

        if dictionary_counter == line_count-1:
            print("Dictionary end is reached! Text will be of smaller size!")
            break

    Dictionary.close()  # закрили

    end_time = time.time()
    return end_time - start_time


def Interface():
    # Вводи
    start_size = 1
    lenth = 1
    Ro_in_main = 4
    Nu_in_main = 3

    for upper in range(10):
        for it in range(10):
            start_size = int(input("Enter the size of starting urn: "))
            if start_size > 0: break
            else: print("The size must be bigger than 0, " + str(10-it) + " attempts remains.")

        for it in range(10):
            lenth = int(input("Enter the text lenth: "))
            if lenth > 0: break
            else: print("The number must be bigger than 0, " + str(10-it) + " attempts remains.")

        for it in range(10):
            Ro_in_main = int(input("Enter Ro: "))
            if Ro_in_main > -1: break
            else: print("Ro must be at least 0, " + str(10-it) + " attempts remains.")

        for it in range(10):
            Nu_in_main = int(input("Enter Nu: "))
            if Nu_in_main > -2: break
            else: print("Nu must be at least than -1, " + str(10-it) + " attempts remains.")

        ret = input("To input again press 2, to continue press 1: ")
        if ret != 2: break

    Urn_main_CREATE(Urn_Main, start_size)                                   # створили основну урну
    print("Processing might take some time...\n")
    get_time = main(Urn_Main,Urn_Txt,lenth,start_size,Ro_in_main,Nu_in_main)    # взяли return-час виконання і зробили шо треба
    print("Finished in " + str(get_time))                                   # Час

    for iterable in range(15):
        ch = int(input("To view created txt array press 1\nTo viev tokens press 2\nTo get .txt file press 3\nElse to exit "))

        if ch == 1:                 # сирий масив слів
            print(Urn_Txt)
            print('Len of txt:'+str(len(Urn_Txt)))
            print('\n-----')
            any = int(input("To choose again press 1, else - exit: "))
            if any == 1: continue
            if any != 1: quit()


        if ch == 2:                 # токени начастіше->найрідше
            for i in Urn_Txt:
                if i not in Out_Tokens:
                    Out_Tokens[i] = 1
                else:
                    Out_Tokens[i] += 1

            sorted_keys = sorted(Out_Tokens, key=Out_Tokens.get, reverse=True)
            for w in sorted_keys:
                Sorted_Tokens[w] = Out_Tokens[w]
            print(Sorted_Tokens)
            print("Number of words = "+ str(len(Urn_Txt)))
            print('\n-----')
            any = int(input("To choose again press 1, else - exit: "))
            if any == 1: continue
            if any != 1: quit()


        if ch == 3:                   # для файла
            Created_Text = open('Created_Text.txt', 'w')
            txt_str = ""
            for i in Urn_Txt:
                txt_str += i
                txt_str += ' '
            Created_Text.write(txt_str)
            Created_Text.close()
            print('Created!\n-----')
            any = int(input("To choose again press 1, else - exit: "))
            if any == 1: continue
            if any != 1: quit()


Interface()