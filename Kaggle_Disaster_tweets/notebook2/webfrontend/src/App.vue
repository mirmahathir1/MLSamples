<template>
  <v-app>
    <v-app-bar
      app
      color="primary"
      dark
    >
      <v-toolbar-title>Disaster Tweets</v-toolbar-title>
<!--      <div class="d-flex align-center">-->
<!--        <v-img-->
<!--          alt="Vuetify Logo"-->
<!--          class="shrink mr-2"-->
<!--          contain-->
<!--          src="https://cdn.vuetifyjs.com/images/logos/vuetify-logo-dark.png"-->
<!--          transition="scale-transition"-->
<!--          width="40"-->
<!--        />-->

<!--        <v-img-->
<!--          alt="Vuetify Name"-->
<!--          class="shrink mt-1 hidden-sm-and-down"-->
<!--          contain-->
<!--          min-width="100"-->
<!--          src="https://cdn.vuetifyjs.com/images/logos/vuetify-name-dark.png"-->
<!--          width="100"-->
<!--        />-->
<!--      </div>-->

      <v-spacer></v-spacer>

<!--      <v-btn-->
<!--        href="https://github.com/vuetifyjs/vuetify/releases/latest"-->
<!--        target="_blank"-->
<!--        text-->
<!--      >-->
<!--        <span class="mr-2">Latest Release</span>-->
<!--        <v-icon>mdi-open-in-new</v-icon>-->
<!--      </v-btn>-->
    </v-app-bar>

    <v-content>
      <v-row>
        <v-col sm="3" cols="12">
        </v-col>
        <v-col sm="6" cols="12">
          <v-sheet elevation="12" class="pa-5 text-center">
            <v-textarea
                    outlined
                    v-model="model"
                    :auto-grow="autoGrow"
                    :clearable="clearable"
                    :counter="counter ? counter : false"
                    :filled="filled"
                    :flat="flat"
                    :hint="hint"
                    :label="label"
                    :loading="loading"
                    :no-resize="noResize"
                    :outlined="outlined"
                    :persistent-hint="persistentHint"
                    :placeholder="placeholder"
                    :rounded="rounded"
                    :row-height="rowHeight"
                    :rows="rows"
                    :shaped="shaped"
                    :single-line="singleLine"
                    :solo="solo"
            ></v-textarea>
            <div class="my-2">
              <v-btn color="primary" :loading="fetching" @click="buttonPressed" rounded>Predict</v-btn>
            </div>
            <v-alert
                    v-if="text.length!==0"
                    dense
                    text
                    :type="type"
                    class="mt-2 text-center"
            >
              Value: {{ text }}
            </v-alert>
          </v-sheet>
        </v-col>
        <v-col sm="3" cols="12">
        </v-col>
      </v-row>
    </v-content>
  </v-app>
</template>

<script>
  import axios from 'axios';
export default {
  name: 'App',

  components: {

  },

  data: () => ({
    autoGrow: false,
    autofocus: true,
    clearable: false,
    counter: 0,
    filled: false,
    flat: false,
    hint: '',
    label: '',
    loading: false,
    model: null,
    noResize: false,
    outlined: false,
    persistentHint: false,
    placeholder: 'Write a tweet here',
    rounded: false,
    rowHeight: 24,
    rows: 1,
    shaped: true,
    singleLine: false,
    solo: false,

    fetching: false,
    text:"",
    type:"success",
  }),
  methods:{
    buttonPressed(){
      this.type="warning"
      this.text="Fetching prediction...";
      this.fetching=true;
      console.log("Button Pressed. Text:",this.model);
      // 'http://192.168.1.35:8000/disaster'
      // https://thawing-coast-54297.herokuapp.com/
      axios.post('http://192.168.1.35:8000/disaster', {
        mytext: this.model,
      }).then((response) => {
        console.log(response.data);
        if(response.data.prediction=="1"){
          this.type="error";
          this.text="This is a text about a disaster !";
        }else{
          this.type="success";
          this.text="This is a normal text !"
        }
      }).catch((error) => {
        //console.log(error);
        this.type="warning"
        this.text=error;
      }).finally(()=>{
        this.fetching=false;
      });
    }
  }
};
</script>
