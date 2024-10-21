# КОМП'ЮТЕРНА ЛІНГВІСТИКА ТА ОБРОБКА ПРИРОДНОЇ МОВИ

## Структура курсу

**Порядок лабораторних**:

[стара нумерація, актуальна для 1го семестру 2024&ndash;2025 навчального року, в дужках значення тих самих робіт у новій нумерації]

__ФЕС3__: 01 (01), 02 (02), 04 (04), 05 (05), 08а (07б), 09 (14), 10 (15), 13 (21), 15 (22), 16 (23)

[нова нумерація, яка використовується в цьому репозиторії]

__ФЕС-3__: 01, 02, 04, 07б, 14, 15, 21, 22, 23, 26

## Структура репозиторію

- папка `images`: у ній містяться файли, потрібні для створення docker-зображень (Dockerfile та `build`-скрипт), а також файли, потрібні для запуску програм у docker-контейнерах (`run`-шаблон і `run`-скрипт):

    - папка `nlp-wine`: містить файли для запуску `.exe`-файлів через docker-контейнер.  
    `run.sh` тут запускає не команду `docker run [OPTIONS]` напряму, а за допомогою проміжного скрипта `docker-wine`, взятого з [однойменного репозиторію](https://github.com/scottyhardy/docker-wine) (який ми використовуємо як baseimage для `nlp-wine`) авторства [scottyhardy](https://github.com/scottyhardy).

    - папка `nlp-java`: містить файли для запуску `.jar`-файлів через docker-контейнер.

    - папка `nlp-python`: містить файли для запуску `.py`-файлів через docker-контейнер.

- папка `programs`: містить файли програм:

    - папка `binaries`: містить програми в бінарному вигляді.

    - папка `jars`: містить програми у вигляді `.jar`-файлів.

    - папка `python`: містить програми у вигляді `.py`-скриптів.

- папка `scripts`: у ній містяться скрипти для запуску програм у зображеннях (переважно ці скрипти складаються з одної команди для запуску програми в контейнері).

- папка `tasks`: у ній містяться описи завдань для лабораторних.  
Це продубльовані завдання лабораторних з матеріалів на Google Drive з поправкою на те, що ви запускатимете їх за допомогою Docker.

- файл `build.sh`: скрипт, який використовується для створення docker-зображення (якщо воно не створене) та формування.

- файл `lab_report_template.tex` --- шаблон для звітів у $\LaTeX$.

## Перевірені системи

| СИСТЕМА    | ВЕРСІЯ   | АРХІТЕКТУРА    |
| ---------- | -------- | -------------- |
| MANJARO    | 6.5      | `amd64/x86-64` |
| macOS      | 14.1     | `amd64/x86-64` |


## Користування репозиторієм

### Linux

1. Встановіть Docker Engine, [вибравши свій дистрибутив](https://docs.docker.com/engine/install/) на офіційному сайті Docker.

2. Скачати відповідний архів з docker-зображенням [тут].  
Трохи складніший варіант: збудувати самостійно зображення з Dockerfile, користуючись скриптом build.sh, який знаходиться в репозиторії, в папці з відповідним зображенням.
В такому випадку ви пропускаєте пункт 3 і одразу переходите до пункту 4.

3. Завантажити архів як зображення в Docker за допомогою команди `docker load`:  
```
docker load -i <path to image tar file>
```

4. Запустіть скрипт build.sh, який знаходиться в корені репозиторію, вказавши назву зображення, яке ви хочете скласти, тобто:
```
./build.sh wine
```
або
```
./build.sh java
```
**Важливо** запускати цей скрипт з кореня репозиторію.
Цей скрипт перевірить, чи вже існує потрібне зображення, якщо ні, то він збудує його на основі Dockerfile, якщо так --- зґенерує скрипт run.sh у папці зображення (тобто в `images/wine` або `images/java`) з вказаними шляхами до потрібних папок у системі.  
До них входять:
    - шлях до репозиторію (змінна `NLP_HOME`), з якого виводяться шляхи до папки з програмами (змінні `LOCAL_JARS` і `LOCAL_BINS`) і папки зі скриптами (змінна `LOCAL_SCRIPTS`); шлях до папки з даними (змінна `NLP_DATA`);
    
    - шлях до папки, в яку ви хочете зберігати результати (змінна `NLP_RESULTS`).

`NLP_HOME` визначається автоматично, як корінь скачаного репозиторію. `NLP_DATA` і `NLP_RESULTS` вводяться через термінал під час виконання скрипта.  
За замовчуванням папки `LOCAL_JARS`/`LOCAL_BINS`, `LOCAL_SCRIPTS`, `NLP_DATA` і `NLP_RESULTS` монтуватимуться в папку `/mnt` в корені файлової системи зображення.
За бажанням, це можна змінити, редагуючи змінні `CONT_PROGRAM_HOME`, `CONT_SCRIPT_HOME`, `CONT_RESULT_HOME` і `CONT_DATA_HOME` в скрипті run.sh.

5. У корені репозиторію в папці scripts (туди веде шлях, який міститься у змінній `LOCAL_SCRIPTS`) знаходяться скрипти, назва кожного з яких відповідає назві певної програми з курсу.
Щоб вибрати програму, яку ви хочете запустити, в скрипті run.sh потрібно вказати назву цього скрипта.
**Зауважте, що скрипт буде працювати тільки за умови, що вони не змінювали шляхи у змінні `CONT_PROGRAM_HOME`.**
В іншому випадку вам потрібно в скрипті run.sh вказувати новий шлях до файлу виконання програми.

6. Запустіть контейнер і програму за допомогою скрипту run.sh таким чином:
```
./images/[IMAGE]/run.sh [SCRIPT_NAME]
```
Наприклад:
```
./images/wine/run.sh ngram
```
або
```
./images/java/run.sh repetition_parameter
```
або
```
./images/python/run.sh FAR
```

7. Збережіть результати розрахунків у папку `/mnt/results/`, або будь-яку іншу, яку ви примонтували до контейнера ззовні, користуючись графічним інтерфейсом програми, якою ви користуєтесь (усі програми дозволяють вручну вибрати потрібний шлях).

8. Після того, як ви закриєте вікно програми, контейнер автоматично закриється, збережуться лише ті файли, що знаходяться в примонтованих ззовні папках.

## Виправлення проблем

- якщо ви зіткнулися з помилкою на зразок: `xhost: authorization required, but no authorization protocol specified`:
    - якщо у вас macOS, то спершу перевірте, чи у вас дозволені підключення від мережевих клієнтів у налаштуваннях XQuartz: перейдіть у вкладку `Security` і подивіться, чи у вас ввімкнене налаштування `Allow connections from network clients`;
    - у відповідному `run`-скрипті змініть рядки `xhost +local:root` та `xhost -local:root` на `xhost +127.0.0.1` та `xhost -127.0.0.1` відповідно.

- перед викликом скрипту `2part_splitter.sh` (для `nlp-python`) виконайте команду `xhost+`, а після завершення роботи з ним --- команду `xhost-`.
