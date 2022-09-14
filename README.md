# S1 Coastline Change

## Installation

### Windows

1. Install `pipwin` using following command:

    ```terminal
    > pip install pipwin
    ```

2. Install `GDAL` and `Fiona` using `pipwin`:

    ```terminal
    > pipwin refresh
    > pipwin install GDAL
    > pipwin install Fiona
    ```

3. Install other depedencies in `requirements.txt`:

    ```terminal
    > pip install -r requirements.txt
    ```

## Get `PC_SDK_SUBSCRIPTION_KEY`

1. Open Planetary Computer Hub and sign in with the Microsoft Accout
2. Open terminal and type the following command:

    ```terminal
    > echo $PC_SDK_SUBSCRIPTION_KEY
    ```

3. Copy the key
4. In local machine, open terminal type the following command:

    ```terminal
    > planetarycomputer configure
    ```

5. Paste the key into the prompt.
