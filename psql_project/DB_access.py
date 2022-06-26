#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Aleksander Ratajczyk
# Created Date: 23.06.2022
# email       : ale.ratajczyk@gmail.com
# status      : Prototype, next versions in deveopment
# git         : https://github.com/Aleksanderro/Data-Science-Projects

# version = 1.0
# ----------------------------------------------------------------------------
""" Project consists of icon file, text file (not attached, contain private password to database) and python script
file. App allows to connect to the sql (PostreSQL) database and perform queries in it. Everything is secured by
connection status checking and performing rollbacks if any operation may be dangerous to the database. Project uses
only two libraries - Psycopg2 for psql operations and tkinter as GUI.
The structure of app is simple - encapsulate  things with Object Oriented Programming using classes:
- "from_sql" to connect to database, create database, perform queries and catch errors,
- "App" to create GUI and manage logic.
The goal was to create plain GUI with a few buttons and commands with text feedback to manage if operation was
successful or incorrect. More specified info is still manageable in terminal as printing is very easy and powerful
debugging tool.
Database app is my second big project. I hope the code is readable and allows to help someone with vision of using sql
in Python. """

import tkinter as tk

import psycopg2
from psycopg2 import Error

class from_sql:
    """Manage connection to database, queries and backup"""

    def __init__(self, db_="food_test", user_="postgres", host_="localhost", port_=5432):
        """Init connection to the database"""

        self.password = ""
        with open("password.txt") as f:  # get password from file
            self.password = f.readline()

        try:
            # establish connection
            self.connection = psycopg2.connect(database=db_,
                                               user=user_,
                                               password=self.password,  # get from file
                                               host=host_,
                                               port=port_)
            self.connection.autocommit = True

            # create a cursor object
            self.cursor = self.connection.cursor()

            print(f"Connection status:")
            # attribute "closed = 0" is equal to open connection
            if self.connection.closed == 0 and self.cursor.closed == 0:
                print("Opened\n")

        # catch potential errors
        except psycopg2.InterfaceError as err:
            print(f"Interface error: {err}\n")

        except psycopg2.OperationalError as err:
            print(f"Operational error: {err}\n")

        except psycopg2.ProgrammingError as err:
            print(f"Programming error: {err}\n")

        except (Exception, Error) as err:
            print(f"Other error: {err}\n")

    def Reconnect(self, db_="food_test", user_="postgres", host_="localhost", port_=5432):
        """Reconnect to the database if connections is lost"""

        # establish connection
        print("\n Connection lost")
        try:
            self.connection = psycopg2.connect(database=db_,
                                               user=user_,
                                               password=self.password,  # get from file
                                               host=host_,
                                               port=port_)

            self.connection.autocommit = True

            # create a cursor object
            self.cursor = self.connection.cursor()

            if self.connection.closed == 0 and self.cursor.closed == 0:
                print("Reconnected successfully\n")

        # catch errors
        except psycopg2.InterfaceError as err:
            print(f"Interface error: {err}\n")

        except psycopg2.OperationalError as err:
            print(f"Operational error: {err}\n")

        except psycopg2.ProgrammingError as err:
            print(f"Programming error: {err}\n")

        except (Exception, Error) as err:
            print(f"Other error: {err}\n")

    def __enter__(self):
        """1. magic method allowing to use 'with' statement"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """2. magic method allowing to use 'with' statement"""
        self.close()

    def Connection(self, extended_info=None):
        """Return connection"""

        if self.connection.closed != 0:
            print("Connection closed.")

        # additional parameter to see extended info about connection
        if extended_info:
            return self.connection
        else:
            # 0 for open
            return self.connection.closed

    def Cursor(self, extended_info=None):
        """Return cursor"""

        if self.cursor.closed != 0:
            print("Cursor closed.")

        # additional parameter to see extended info about cursor
        if extended_info:
            return self.cursor
        else:
            # 0 for open
            return self.cursor.closed

    def Query(self, query, *args):
        """Execute a command and return its content"""

        # Check connecion status if lost then reconnect automatically
        if self.Connection() != 0 or self.Cursor() != 0:
            self.Reconnect()

        try:
            self.cursor.execute(query, args)
            print("Executing command")
            # return fetched query to be able to print it
            return self.cursor.fetchall()

        # catch potential errors and return them to manage error handling in functions
        except psycopg2.InterfaceError as err:
            print(f"Interface error: {err}\n")
            return "error", err

        except psycopg2.OperationalError as err:
            print(f"Operational error: {err}\n")
            return "error", err

        except psycopg2.ProgrammingError as err:
            print(f"Programming error: {err}\n")
            return "error", err

        except (Exception, Error) as err:
            print(f"Other error: {err}\n")
            return "error", err

    def Disconnect(self):
        """Close connection to the db"""
        self.connection.close()
        self.cursor.close()
        print(f"Connection closed")

    def CreateBackup(self, print_flag=None):
        """Create local backup of tables from database, return tables names and tables contents"""
        with self.connection:
            with self.cursor:
                print(f"Creating copy of existing database.\n")

                ### MEAL TABLE
                # Create a query to show meal table
                meal_table_query = "SELECT * FROM meal"
                # Execute query
                self.cursor.execute(meal_table_query)
                # Fetch data to create backup of meal table
                meal_backup = self.cursor.fetchall()

                if print_flag:
                    # print data for meal table
                    print(f"Id \t name \t\t\t\t description")
                    print("----------------------------------------------------------------")
                    for row in meal_backup:
                        print(f"{row[0]} \t {row[1]} \t {row[2]}")

                ### INGREDIENT TABLE
                # Create a query to show ingredient table
                ingredient_table_query = "SELECT * FROM ingredient"
                # Execute query
                self.cursor.execute(ingredient_table_query)
                # Fetch data to create backup of ingredient table
                ingredient_backup = self.cursor.fetchall()

                if print_flag:
                    # print data for meal table
                    print(f"\nId \t name")
                    print("----------------------------------------------------------------")
                    for row in ingredient_backup:
                        print(f"{row[0]} \t {row[1]}")

                ### NUTRITIENTS TABLE
                # Create a query to show nutritients table
                nutritients_table_query = "SELECT * FROM nutritients"
                # Execute query
                self.cursor.execute(nutritients_table_query)
                # Fetch data to create backup of nutritients table
                nutritients_backup = self.cursor.fetchall()

                if print_flag:
                    # print data for meal table
                    print(f"\nId \t calories \t protein \t carbohydrate \t fat")
                    print("----------------------------------------------------------------")
                    for row in nutritients_backup:
                        print(f"{row[0]} \t {row[1]} \t\t {row[2]} \t\t {row[3]} \t\t\t {row[4]}")

                ### RECIPE TABLE
                # Create a query to show nutritients table
                recipe_table_query = "SELECT * FROM recipe"
                # Execute query
                self.cursor.execute(recipe_table_query)
                # Fetch data to create backup of recipe table
                recipe_backup = self.cursor.fetchall()

                if print_flag:
                    # print data for meal table
                    print(f"\nrecipe_id \t meal_id \t ingredient_id \t nutritients_id")
                    print("----------------------------------------------------------------")
                    for row in recipe_backup:
                        print(f"{row[0]} \t\t\t {row[1]} \t\t\t {row[2]} \t\t\t\t {row[3]}")

                # list tables names
                tables_names_query = "SELECT * FROM information_schema.tables WHERE table_schema = 'public'"
                self.cursor.execute(tables_names_query)
                # expanded list of all columns in listed schema information
                tables_names_expanded = self.cursor.fetchall()
                tables_names = []

                for row_ in tables_names_expanded:
                    tables_names.append(row_[2])  # appending only tables names to container
                    # print(f"{row_[2]}")

                # create container to store backup tables
                table_container = [meal_backup, ingredient_backup, nutritients_backup, recipe_backup]

                return table_container, tables_names


class App(tk.Tk):
    """Main app class to manage GUI and majority of logic"""

    def __init__(self, *args, **kwargs):
        """Init function to create main window and class variables"""

        tk.Tk.__init__(self, *args, **kwargs)

        # Main window config
        self.geometry("700x500")
        self.title("Diet Database")
        self.config(bg="floral white")
        self.iconbitmap("icon.ico")
        self.attributes("-topmost", 1)  # topmost window in windows explorer
        self.resizable(False, False)  # non resizable

        # Initialize the instance of "from_sql" object to share it within class "App"
        self.database = from_sql()
        self.tables_content, self.tables_names = self.database.CreateBackup()

        # list without "recipe" table
        self.tables_names_short = self.tables_names.copy()
        self.tables_names_short.pop(1)

        # dictionary storing tables names as keys and columns names as values
        self.db_structure = self.get_column_identifier()

        # variables for "next_table_btn" widget and "next_table" function
        self.next_table_iter = 0  # iterator to loop through tables names in funcion "next table"
        self.table_name = tk.StringVar()  # StringVar to set dynamically name of the table
        self.table_name.set(self.tables_names[0])  # setting starting name to first element of tables names

        # next table flag allows or prevent using "next_table" function
        self.next_table_flag = True

        # variables for "column_name_lbl" widget and "add_entry" function
        self.add_entry_iter = 0
        self.add_entry_values = []
        self.column_iden = tk.StringVar()
        self.column_iden.set(self.db_structure[self.tables_names[0]][self.add_entry_iter])

        self.adding_record_var = tk.StringVar()  # show if adding record was succesfull

        # show errors and its contents within command section
        self.error_cmd = tk.StringVar()
        self.error_cmd_message = tk.StringVar()

        # connection and cursor dynamic status
        self.connection_status_var = tk.StringVar()
        self.cursor_status_var = tk.StringVar()

        # commands list variable and names
        self.chosen_command_var = tk.IntVar()
        self.command_list_names = ['show meals', 'show detailed meals', 'show ingredients', 'your command (type)']

        # tables_content -> stores content of all tables
        # tables_names -> stores names of 4 tables
        # tables_names_short -> stores names of 3 tables (without recipe)

        # call GUI to create Tkinter widgets
        self.GUI()

        # call function to check connection and cursor
        self.check_conn_and_curr()

    def GUI(self):
        """Create widgets and show them on main window"""

        ### Frames
        # Record frame
        r_frame = tk.Frame(self)
        r_frame.config(bg="floral white")
        r_frame.place(x=0, y=0)

        # Command frame
        c_frame = tk.Frame(self)
        c_frame.config(bg="floral white")
        """c_frame.pack(expand=True, anchor=tk.W)"""
        c_frame.place(x=0, y=200)

        # Command list frame
        cl_frame = tk.Frame(self)
        cl_frame.config(bg="floral white")
        """cl_frame.pack(anchor=tk.N)"""
        cl_frame.place(x=500, y=0)

        # Utility frame
        ut_frame = tk.Frame(self)
        ut_frame.config(bg="floral white")
        """ut_frame.pack(expand=True, anchor=tk.SE, ipadx=5, ipady=5)"""
        ut_frame.place(x=0, y=455, width=695)

        ### Record section
        # ---row 0
        self.column_name_lbl = tk.Label(r_frame,  # dynamic
                                        textvariable=self.column_iden,
                                        width=15,
                                        font=("Arial", 16),
                                        bg="floral white")
        self.column_name_lbl.grid(row=0, column=1, padx=5, pady=5)

        # ---row 1
        self.add_record_lbl = tk.Label(r_frame,  # static
                                       text="add record:",
                                       font=("Arial", 20),
                                       bg="floral white")
        self.add_record_lbl.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        self.add_record_entry = tk.Entry(r_frame, bd=2)
        self.add_record_entry.grid(row=1, column=1, padx=5, pady=5)

        self.add_record_btn = tk.Button(r_frame,
                                        text="Add",
                                        font=("Arial", 16),
                                        width=5,
                                        bg="bisque3",
                                        command=self.add_entry)
        self.add_record_btn.grid(row=1, column=2, padx=5, pady=5)

        # ---row 2
        self.path_info_lbl = tk.Label(r_frame,  # static
                                      text="Table:",
                                      font=("Arial", 18),
                                      bg="floral white")
        self.path_info_lbl.grid(row=2, column=0, padx=5, pady=5)

        self.path_lbl = tk.Label(r_frame,  # dynamic
                                 textvariable=self.table_name,
                                 font=("Arial", 20),
                                 bg="floral white")
        self.path_lbl.grid(row=2, column=1, padx=5, pady=5)

        self.next_table_btn = tk.Button(r_frame,
                                        text="Next table",
                                        font=("Arial", 13),
                                        bg="bisque3",
                                        command=self.next_table)
        self.next_table_btn.grid(row=2, column=2, padx=5, pady=5)

        # ---row 3
        self.adding_record_lbl = tk.Label(r_frame,
                                          textvariable=self.adding_record_var,
                                          font=("Arial", 16),
                                          bg="floral white")
        self.adding_record_lbl.grid(row=3, column=1, padx=5, pady=5)

        ### Command section
        self.command_lbl = tk.Label(c_frame,  # static
                                    text="command:",
                                    font=("Arial", 20),
                                    bg="floral white")
        self.command_lbl.grid(row=0, column=0, padx=5, pady=5)

        self.command_entry_text = tk.Text(c_frame, bd=2, height=5, width=35)
        self.command_entry_text.grid(row=0, column=1, padx=5, pady=5)

        # ---row 1
        self.run_command_btn = tk.Button(c_frame,
                                         text="Run",
                                         font=("Arial", 16),
                                         width=5,
                                         bg="bisque3",
                                         command=self.execute_command)
        self.run_command_btn.grid(row=1, column=0, padx=5, pady=5)

        self.result_command_lbl = tk.Label(c_frame,  # dynamic
                                           textvariable=self.error_cmd,
                                           font=("Arial", 18),
                                           bg="floral white")
        self.result_command_lbl.grid(row=1, column=1, padx=5, pady=5)

        self.clear_entry_btn = tk.Button(c_frame,
                                         text="Clear",
                                         font=("Arial", 16),
                                         width=5,
                                         bg="bisque3",
                                         command=self.clear_command_entry)
        self.clear_entry_btn.grid(row=1, column=2, padx=5, pady=5)

        # ---row 2
        self.command_error_lbl = tk.Label(c_frame,  # dynamic
                                          textvariable=self.error_cmd_message,
                                          font=("Arial", 14),
                                          bg="floral white")
        self.command_error_lbl.grid(row=2, column=1, padx=5, pady=5)

        ### Command list section
        for indx in range(len(self.command_list_names)):
            self.radio_btn = tk.Radiobutton(cl_frame,
                                            text=self.command_list_names[indx],
                                            variable=self.chosen_command_var,
                                            value=indx,
                                            font=("Arial", 12),
                                            justify="left",
                                            relief=tk.RAISED,
                                            bg="bisque3")

            self.radio_btn.grid(row=indx, column=0)

        ### Utility section
        self.connection_text_lbl = tk.Label(ut_frame,
                                            text="Connection status: ",
                                            font=("Arial", 13),
                                            bg="floral white")
        self.connection_text_lbl.pack(side=tk.LEFT)

        self.connection_status_lbl = tk.Label(ut_frame,
                                              textvariable=self.connection_status_var,
                                              font=("Arial", 13,"bold"),
                                              bg="floral white")
        self.connection_status_lbl.pack(side=tk.LEFT,padx=10)

        self.cursor_text_lbl = tk.Label(ut_frame,
                                        text="cursor status: ",
                                        font=("Arial", 13),
                                        bg="floral white")
        self.cursor_text_lbl.pack(side=tk.LEFT,padx=10)

        self.cursor_status_lbl = tk.Label(ut_frame,
                                          textvariable=self.cursor_status_var,
                                          font=("Arial", 13,"bold"),
                                          bg="floral white")
        self.cursor_status_lbl.pack(side=tk.LEFT)

        self.quit_btn = tk.Button(ut_frame,
                                  text="quit",
                                  font=("Arial", 16),
                                  width=5,
                                  bg="bisque3",
                                  command=self.quit_app)
        self.quit_btn.pack(anchor=tk.E, side=tk.RIGHT,fill='both')

    def get_column_identifier(self):
        """Make queries, return dictionary with tables names as keys and columns names as its values."""

        query = "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = " \
                "%s AND column_name NOT LIKE '%%_id'"

        query_dict = {}

        for query_var in self.tables_names_short:  # loop through tables names
            query_list = []  # empty list to store table name
            query_result = self.database.Query(query, query_var)

            for itr in query_result:  # loop through columns names of each table
                tup_to_str = ''.join(itr)  # transform tuple object to string
                query_list.append(tup_to_str)

            query_dict[query_var] = query_list  # bind table name with list of its column names

        print("Above queries are from 'get_column_identifier' function.")
        print(f"Dict is: {query_dict}\n")
        return query_dict

    def check_conn_and_curr(self):
        """Set connection and cursor labels based on their status."""

        conn = self.database.Connection()

        if conn == 0:
            self.connection_status_var.set("open")
        elif conn != 0:
            self.connection_status_var.set("closed")

        curr = self.database.Cursor()

        if curr == 0:
            self.cursor_status_var.set("open")
        elif curr != 0:
            self.cursor_status_var.set("closed")

    def add_entry_query(self):
        """Return insert query based on chosen table name."""

        if self.table_name.get() == "ingredient":
            query = "INSERT INTO ingredient (ingredient_name) VALUES (%s) RETURNING ingredient_name"
            query_params = tuple(self.add_entry_values)

        elif self.table_name.get() == "meal":
            query = "INSERT INTO meal (meal_name, meal_description) VALUES (%s, %s) RETURNING meal_name"
            print("Got 2 variables here")
            query_params = tuple(self.add_entry_values)

        elif self.table_name.get() == "nutritients":
            query = "INSERT INTO nutritients (calories, protein, carbohydrate, fat) VALUES (%s, %s, %s, %s) RETURNING " \
                    "calories, protein, carbohydrate, fat "
            query_params = tuple(self.add_entry_values)

        # make dynamic label "visible"
        self.adding_record_lbl.config(bg="floral white")

        if "query" and "query_params" in locals():
            query_content = self.database.Query(query, *query_params)  # pass unpacked tuple

            # check if Query returned error
            if query_content[0] == "error":
                print("Error while adding, try again")
                self.adding_record_var.set("Error while adding, try again")

            # no error
            else:
                print("Successfully added")
                self.adding_record_var.set("Successfully added")

        else:
            print("Error while adding, try again")
            self.adding_record_var.set("Error while adding, try again")

        # check status
        self.check_conn_and_curr()

    def add_entry(self):
        """Add identifier based on table and displayed column"""

        actual_table_name = self.tables_names_short[self.next_table_iter]
        column_name = self.db_structure[actual_table_name]

        # check if iter reached last element
        if self.add_entry_iter < len(column_name) - 1:
            self.add_entry_iter += 1
        else:
            self.add_entry_iter = 0

        # set shown column name
        self.column_iden.set(column_name[self.add_entry_iter])

        # get text from entry box
        entry_record = self.add_record_entry.get()
        print(f"Typed: {entry_record}")

        if entry_record != "":  # check if something is typed in entry box
            self.next_table_flag = False  # block next table function

            # add object to temporary list
            self.add_entry_values.append(entry_record)

        self.add_record_entry.delete(0, 'end')  # clear entry box after typing
        # print(self.add_entry_values)

        # check if list holds the same amount of entries as specific table has
        if len(self.add_entry_values) == len(column_name):
            self.next_table_flag = True  # Unlock next table function

            # query to db
            self.add_entry_query()

            self.add_entry_values.clear()

        # check status
        self.check_conn_and_curr()

    def next_table(self):
        """Set name of actual table"""

        self.adding_record_lbl.config(bg="floral white")

        # check if next table flag allows to swap next table
        if self.next_table_flag:
            self.next_table_iter += 1

            if self.next_table_iter == 3:
                self.next_table_iter = 0

            self.table_name.set(self.tables_names_short[self.next_table_iter])
            self.column_iden.set(self.db_structure[self.tables_names_short[self.next_table_iter]][0])
        else:
            print("Function blocked.")

    def quit_app(self):
        """Close connection to database and quit from app."""

        self.database.Disconnect()
        self.destroy()

    def clear_command_entry(self):
        """Clear command entry """

        self.command_entry_text.delete("1.0", tk.END)
        print("Entry cleared")

    def execute_command(self):
        """Execute command based on selected from list"""

        if self.chosen_command_var.get() == 0:  # show meals
            query = "SELECT * FROM meal"

        elif self.chosen_command_var.get() == 1:  # show detailed meals
            query = "SELECT m.meal_name, m.meal_description, i.ingredient_name, n.calories, n.protein, " \
                    "n.carbohydrate, n.fat FROM recipe AS r JOIN meal AS m ON m.meal_id = r.meal_id JOIN ingredient " \
                    "AS i ON i.ingredient_id = r.ingredient_id JOIN nutritients AS n ON n.nutritients_id = " \
                    "r.nutritients_id "

        elif self.chosen_command_var.get() == 2:  # show ingredients
            query = "SELECT * FROM ingredient"

        elif self.chosen_command_var.get() == 3:
            query = self.command_entry_text.get("1.0", "end-1c")  # get value from text box (1st line 1st character
            # and end-newline character"
            print(f"Query: {query}")

        # check if variable "query" exists in local (function) scope
        if 'query' in locals():
            query_content = self.database.Query(query)

            # check if there is any error in commands
            if query_content[0] == "error":

                self.error_cmd.set("Error")
                self.error_cmd_message.set(str(query_content[1]))

                # if query was wrong rollback transaction to clear query cache
                self.database.Connection(True).rollback()

            else:
                # command correct
                self.error_cmd.set("Success")
                self.error_cmd_message.set("")
                self.command_error_lbl.config(bg="floral white")

                for cont in query_content:
                    print(cont)

        else:
            print("Error detected, try again.")

        # check status
        self.check_conn_and_curr()


if __name__ == "__main__":
    app = App()
    app.mainloop()
