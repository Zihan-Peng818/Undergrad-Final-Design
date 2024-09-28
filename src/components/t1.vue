<template>
<el-upload
  class="upload-demo"
  ref="upload1"
  action="http://localhost:5000"
  :on-preview="handlePreview"
  :on-remove="handleRemove"
  :file-list="fileList"
  :auto-upload="false"
  :limit="1"
  list-type="picture"
  :on-success="success">
  <el-button size="small" type="primary">点击上传</el-button>
  <div slot="tip" class="el-upload__tip">请在此处上传变化前的图像</div>
</el-upload>
</template>

<script>
  export default {
    data() {
      return {
        fileList: [],
      };
    },
    methods: {
      success(response, file, fileList){
          console.log(response)
      },
      handleRemove(file, fileList) {
        console.log(file, fileList);
      },
      handlePreview(file) {
        console.log(file);
      },
      submit(){
        this.$refs.upload1.submit();
      }
    }
  }
</script>